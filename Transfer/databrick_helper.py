## ==== Helper function ===
def export_and_get_link(df, file_name, file_format="csv", folder="exports", **kwargs):
    """
    Save a Spark dataframe to DBFS FileStore and return a download link.
    
    Parameters:
        df (DataFrame): Spark DataFrame to save
        file_name (str): Desired file name (e.g., "prediction.csv")
        file_format (str): Format to save ("csv", "parquet", "json")
        folder (str): Folder under FileStore (default = "exports")
        **kwargs: Additional writer options (header=True, delimiter=",", etc.)

    Returns:
        str: Direct browser download link
    """

    # ----------------------------------------------------------------------
    # 1. Build FileStore path
    # ----------------------------------------------------------------------
    dbfs_dir = f"dbfs:/FileStore/{folder}"
    dbutils.fs.mkdirs(dbfs_dir)

    # For formats that generate multiple part files, manage naming separately
    save_path = f"{dbfs_dir}/{file_name}" if file_format != "csv" else f"{dbfs_dir}/{file_name.replace('.csv', '')}"

    # ----------------------------------------------------------------------
    # 2. Write file
    # ----------------------------------------------------------------------
    if file_format == "csv":
        df.coalesce(1).write.mode("overwrite").csv(save_path, **kwargs)
        
        # Get the written part file
        files = dbutils.fs.ls(save_path)
        part_file = [f.path for f in files if f.name.startswith("part-")][0]

        final_path_dbfs = f"dbfs:/FileStore/{folder}/{file_name}"
        
        # Rename part-file → final file name
        dbutils.fs.mv(part_file, final_path_dbfs)
    
    else:
        df.write.mode("overwrite").format(file_format).save(save_path)
        final_path_dbfs = save_path

    # ----------------------------------------------------------------------
    # 3. Build public browser download link
    # ----------------------------------------------------------------------
    instance = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
    
    # FileStore is accessible via /files/
    download_link = f"https://{instance}/files/{folder}/{file_name}"

    return download_link





###================================================================================================================================###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy.stats import shapiro, ks_2samp, zscore
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load cleaned dataset
df = pd.read_csv(r"C:data.csv")

# -----------------------
# 1. Data Audit
# -----------------------
print("Shape:", df.shape)
print(df.info())
print(df.describe(include="all").T)

# Missingness heatmap
msno.heatmap(df)
plt.show()

# Outlier detection: Tukey’s IQR
def detect_outliers_iqr(series):
    q1, q3 = np.percentile(series.dropna(), [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return ((series < lower) | (series > upper)).sum()

outlier_summary = {col: detect_outliers_iqr(df[col]) for col in df.select_dtypes(include=np.number).columns}

# Target balance check
print(df['HH_FLAG'].value_counts(normalize=True))  # assuming HH_FLAG = target

# -----------------------
# 2. Smart Univariate Analysis
# -----------------------
from sklearn.metrics import roc_auc_score

# Continuous vars: distribution + normality
num_cols = df.select_dtypes(include=np.number).drop(columns=['HH_FLAG']).columns
for col in num_cols:
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f"{col} distribution")
    plt.show()
    if df[col].nunique() > 10:
        stat, p = shapiro(df[col].dropna().sample(min(5000, df[col].notna().sum())))
        print(f"{col} Shapiro-Wilk p={p:.4f} {'Normal' if p>0.05 else 'Non-normal'}")

# Categorical vars: Information Value (IV)
def calc_iv(df, feature, target):
    temp = df.groupby(feature)[target].agg(['count','sum'])
    temp['non_event'] = temp['count'] - temp['sum']
    event_rate = temp['sum'].sum()/temp['count'].sum()
    non_event_rate = 1 - event_rate
    temp['event_dist'] = temp['sum']/temp['sum'].sum()
    temp['non_event_dist'] = temp['non_event']/temp['non_event'].sum()
    temp['woe'] = np.log((temp['event_dist']+1e-5)/(temp['non_event_dist']+1e-5))
    iv = ((temp['event_dist']-temp['non_event_dist'])*temp['woe']).sum()
    return iv

iv_scores = {col: calc_iv(df, col, 'HH_FLAG') for col in df.select_dtypes(include='object').columns}
print("Top IV features:", sorted(iv_scores.items(), key=lambda x: x[1], reverse=True)[:10])


## Feature–Target Relationship Deep Dive
from scipy.stats import chi2_contingency
import prince  # for WoE if needed

# Categorical vs Target: Chi-square + Cramer’s V
def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    return np.sqrt(phi2/min(k-1,r-1))

cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    cm = pd.crosstab(df[col], df['HH_FLAG'])
    print(col, "Cramer's V:", cramers_v(cm))

# Continuous vs Target: KS & AUC
for col in num_cols:
    ks, _ = ks_2samp(df.loc[df['HH_FLAG']==1, col].dropna(),
                     df.loc[df['HH_FLAG']==0, col].dropna())
    auc = roc_auc_score(df['HH_FLAG'], df[col].fillna(df[col].median()))
    print(f"{col}: KS={ks:.3f}, AUC={auc:.3f}")

# Binning continuous vars
for col in num_cols:
    df[f"{col}_bin"] = pd.qcut(df[col], q=5, duplicates="drop")
    sns.barplot(x=df[f"{col}_bin"], y=df['HH_FLAG'])
    plt.title(f"{col} vs Target Rate")
    plt.xticks(rotation=45)
    plt.show()


## Multivariate Insights & Interaction Effects
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Correlation + VIF
corr = df[num_cols].corr(method='spearman')
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.show()

vif_data = pd.DataFrame()
vif_data["feature"] = num_cols
vif_data["VIF"] = [variance_inflation_factor(df[num_cols].fillna(0).values, i)
                   for i in range(len(num_cols))]
print(vif_data.sort_values("VIF", ascending=False))

# Interaction effects
for col1 in ['READMIT_FLAG','DIALYSIS_FLAG']:
    for col2 in ['ER_VISIT_COUNT','TOTAL_ENCOUNTERS']:
        cm = pd.crosstab(df[col1], pd.qcut(df[col2], q=5, duplicates="drop"), df['HH_FLAG'], aggfunc='mean').fillna(0)
        sns.heatmap(cm, annot=True, cmap="Blues")
        plt.title(f"{col1} × {col2} vs HH rate")
        plt.show()
## Advanced Explorations (Propensity Focused)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# PCA + clustering
X = df[num_cols].fillna(0)
X_pca = PCA(n_components=2).fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42).fit(X_pca)
df['cluster'] = kmeans.labels_

sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df['cluster'], palette='tab10')
plt.show()

# Compare HH rate across clusters
print(df.groupby('cluster')['HH_FLAG'].mean())

# PSI: Population Stability Index between HH=1 and HH=0
def calc_psi(expected, actual, buckets=10):
    def scale_range(input, buckets):
        return np.linspace(input.min(), input.max(), buckets)
    psi = 0
    breakpoints = scale_range(expected, buckets)
    expected_counts = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_counts = np.histogram(actual, breakpoints)[0] / len(actual)
    for e,a in zip(expected_counts, actual_counts):
        if e>0 and a>0:
            psi += (e-a)*np.log(e/a)
    return psi

psi_results = {}
for col in num_cols:
    psi_results[col] = calc_psi(df.loc[df['HH_FLAG']==0, col].fillna(0),
                                df.loc[df['HH_FLAG']==1, col].fillna(0))
print("Top drift features:", sorted(psi_results.items(), key=lambda x: -x[1])[:10])


Later versions may test advanced imputations (e.g., predictive models or rule-based filling) if they improve recall/precision.
##+===================================================
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

def ks_test_imputation(df, impute_rules, threshold=0.1, plot=False):
    """
    Run KS tests before vs after imputation for numeric features.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe with nulls.
    impute_rules : dict
        Dictionary of {col: strategy} where strategy ∈ {"zero", "median", "mean"}.
    threshold : float
        KS threshold above which drift is flagged.
    plot : bool
        If True, plots before vs after distributions for flagged cols.

    Returns
    -------
    ks_results : pd.DataFrame
        Summary table with KS statistics and drift flags.
    df_imputed : pd.DataFrame
        Dataframe with imputed values.
    """

    df_imputed = df.copy()
    results = []

    for col, strategy in impute_rules.items():
        if col not in df.columns:
            continue

        series_before = df[col].dropna()

        # ---- Edge cases ----
        if series_before.empty:  # all NaN
            df_imputed[col] = 0
            results.append((col, np.nan, "ALL_NAN"))
            continue
        if df[col].nunique(dropna=True) == 1:  # constant col
            df_imputed[col].fillna(df[col].mode()[0], inplace=True)
            results.append((col, 0.0, "CONSTANT"))
            continue

        # ---- Imputation ----
        if strategy == "zero":
            fill_val = 0
        elif strategy == "median":
            fill_val = df[col].median()
        elif strategy == "mean":
            fill_val = df[col].mean()
        else:
            raise ValueError(f"Unknown strategy {strategy} for {col}")

        df_imputed[col].fillna(fill_val, inplace=True)

        # ---- KS test ----
        series_after = df_imputed[col]
        ks_stat, p_val = ks_2samp(series_before, series_after)

        results.append((col, ks_stat, "DRIFT" if ks_stat > threshold else "OK"))

        # ---- Optional plot ----
        if plot and ks_stat > threshold:
            plt.figure(figsize=(6,4))
            series_before.hist(alpha=0.5, bins=30, label="Before")
            series_after.hist(alpha=0.5, bins=30, label="After")
            plt.title(f"{col}: KS={ks_stat:.3f}")
            plt.legend()
            plt.show()

    ks_results = pd.DataFrame(results, columns=["column", "ks_stat", "status"])
    return ks_results, df_imputed


##===============
import numpy as np
import pandas as pd

# df_local: a pandas DataFrame you exported from Snowflake (CSV/Parquet/etc.)
target_col    = "HH_FLAG"   # binary target
pos_label     = 1
rare_min_prop = 0.01
eps           = 1e-6

def calc_iv_woe_pandas(df, feature, target, pos_label=1, rare_min_prop=0.01, eps=1e-6):
    s = df[feature].astype("object").fillna("__MISSING__")
    y = (df[target] == pos_label).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return 0.0, pd.DataFrame(columns=["level","n","events","nonevents","dist_bad","dist_good","woe","iv_contrib"]), {}

    ct = pd.crosstab(s, y)
    if 0 not in ct.columns: ct[0] = 0
    if 1 not in ct.columns: ct[1] = 0
    ct = ct[[0,1]]

    n = ct.sum(axis=1)
    rare = n / n.sum() < rare_min_prop
    if rare.any():
        other = ct.loc[rare].sum()
        ct = ct.loc[~rare].copy()
        ct.loc["__OTHER__"] = other

    nonevents, events = ct[0], ct[1]
    dist_good = (nonevents + eps) / (nonevents.sum() + eps)  # non-events
    dist_bad  = (events    + eps) / (events.sum()    + eps)  # events

    woe = np.log(dist_bad / dist_good)
    iv_contrib = (dist_bad - dist_good) * woe
    iv_total = float(iv_contrib.sum())

    woe_df = pd.DataFrame({
        "level": ct.index,
        "n": (nonevents + events).values,
        "events": events.values,
        "nonevents": nonevents.values,
        "dist_bad": dist_bad.values,
        "dist_good": dist_good.values,
        "woe": woe.values,
        "iv_contrib": iv_contrib.values
    }).sort_values("iv_contrib", ascending=False).reset_index(drop=True)

    return iv_total, woe_df, dict(zip(woe_df["level"], woe_df["woe"]))

# --- run for many categoricals ---
exclude = {"PAT_BIRTH_DT", target_col}
categorical_cols = (
    df_local
      .select_dtypes(include=["object","category"])
      .columns.difference(exclude)
      .tolist()
)

rows, wmaps = [], {}
for c in categorical_cols:
    iv, wtbl, wmap = calc_iv_woe_pandas(df_local, c, target_col, pos_label, rare_min_prop, eps)
    rows.append({"Feature": c, "IV": iv})
    wmaps[c] = wmap

iv_df = pd.DataFrame(rows).sort_values("IV", ascending=False).reset_index(drop=True)
print(iv_df.head(10))

# optional: create _WOE columns for top K
K = 10
df_local_woe = df_local.copy()
for c in iv_df.head(K)["Feature"]:
    m = wmaps[c]
    s = df_local_woe[c].astype("object").fillna("__MISSING__")
    if "__OTHER__" in m:
        s = s.where(s.isin(m.keys()), "__OTHER__")
    df_local_woe[c + "_WOE"] = s.map(m)



##====================
Day 1 – Data Audit + Smart Univariate Analysis
	• Data Audit
		○ Data types, missingness heatmap, missingness dependency (Missingno, Little’s MCAR test).
		○ Outlier detection with Tukey’s IQR & z-scores.
		○ Target balance check (HH vs non-HH).
	• Smart Univariate Analysis
		○ For continuous vars: Distribution + normality check (Shapiro/K-S test).
		○ For categorical vars: Entropy/Information Value (IV) against target.
		○ Outlier impact: Winsorization vs log-transform candidates flagged.
	• Deliverable: Missingness/Outlier treatment strategy, Top 10 strongest features by univariate predictive power (IV, correlation with target).

Day 2 – Feature-Target Relationship Deep Dive
	• Categorical vs Target
		○ Chi-square with Cramer’s V effect size.
		○ Weight of Evidence (WoE) transformation for high-cardinality features.
	• Continuous vs Target
		○ KS statistic / AUC measure per feature.
		○ Bin continuous features (quantiles) → compare target rate trends.
		○ Partial dependency (1-D plots).
	• Deliverable: Ranked list of features most associated with HH adoption, with plots showing monotonic/non-monotonic trends.

Day 3 – Multivariate Insights & Interaction Effects
	• Correlation & Multicollinearity
		○ Pearson/Spearman matrix → Variance Inflation Factor (VIF).
		○ Remove redundant features (>0.8 correlation).
	• Interactions
		○ Chi-square automatic interaction detection (CHAID) or decision tree splits on target.
		○ 2-way interaction heatmaps (e.g., chronic condition × visit frequency).
	• Deliverable: List of interaction candidates + reduced feature set with redundancy handled.

Day 4 – Advanced Explorations (Propensity Focused)
	• Clustering / Segmentation
		○ Use PCA/UMAP + k-means/HDBSCAN to see if natural groups align with HH takers.
		○ Compare HH rate across clusters.
	• Propensity Features Diagnostics
		○ Stability check (PSI – Population Stability Index) between HH=1 and HH=0 groups.
		○ Feature drift: Are some features significantly different distributions between groups?
Deliverable: Segment insights (e.g., “Cluster 3 = older, frequent PCO visitors, 3x more likely to take HH”).



### ==========================================================================================================


from pyspark.sql.functions import col, when, isnan

# 1. Get column type lists
numeric_cols_all = [f.name for f in trainingdf.schema.fields if "StringType" not in str(f.dataType)]
categorical_cols_all = [f.name for f in trainingdf.schema.fields if "StringType" in str(f.dataType)]

# 2. Clean numeric columns (replace null/NaN with 0)
for c in numeric_cols_all:
    trainingdf = trainingdf.withColumn(
        c, when(isnan(col(c)) | col(c).isNull(), 0).otherwise(col(c))
    )

# 3. Clean categorical columns (replace null with 'UNK')
trainingdf = trainingdf.fillna('UNK', subset=categorical_cols_all)

# 4. Final sanity check: print any remaining null/NaN counts
print("\n=== Final Data Check ===")
for c in numeric_cols_all:
    bad_count = trainingdf.filter(col(c).isNull() | isnan(col(c))).count()
    if bad_count > 0:
        print(f"⚠ {c} → {bad_count} bad values")
for c in categorical_cols_all:
    bad_count = trainingdf.filter(col(c).isNull()).count()
    if bad_count > 0:
        print(f"⚠ {c} → {bad_count} nulls")
print("=== Check Complete ===")

##=======================
import numpy as np
import pandas as pd
from evidently.report import Report

def drift_table_from_report(rep: Report) -> pd.DataFrame:
    res = rep.as_dict()
    rows = []

    for sec in res.get("metrics", []):
        result = sec.get("result", {}) or {}
        # 0.4.x uses "drift_by_columns"; some versions expose different keys
        by_cols = (result.get("drift_by_columns")
                   or result.get("columns")
                   or {})

        for col, info in (by_cols or {}).items():
            # p_value can be float/None/"N/A"
            p = info.get("p_value", None)
            try:
                p = float(p)
            except (TypeError, ValueError):
                p = np.nan

            # drift_score can be float/None/dict
            ds = info.get("drift_score", None)
            try:
                ds = float(ds)
            except (TypeError, ValueError):
                ds = np.nan

            # name may live in different places across versions
            stat_name = (
                info.get("stattest_name")
                or (info.get("stattest", {}) or {}).get("name")
                or None
            )

            dd = info.get("drift_detected", None)
            if dd is not None:
                dd = bool(dd)

            rows.append({
                "column": col,
                "stattest_name": stat_name,
                "p_value": p,
                "drift_score": ds,
                "drift_detected": dd,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Sort safely; push NaNs to bottom
    return df.sort_values(["drift_detected", "p_value"],
                          ascending=[False, True],
                          na_position="last")
