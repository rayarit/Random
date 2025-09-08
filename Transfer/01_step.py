##------------------------------
## PCO-HH baseline traning Code 
##------------------------------

## Step 1 
import time
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np

# Base logistic regression
lr = LogisticRegression(
    max_iter=1000,
    solver="lbfgs",
    class_weight="balanced",
    n_jobs=-1
)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', lr)
])

# RFECV with timing
start_time = time.time()

rfecv = RFECV(
    estimator=pipe,
    step=1,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring='roc_auc',
    n_jobs=1,  # keep low if memory issues
    importance_getter=lambda est: est.named_steps['logreg'].coef_
)

rfecv.fit(X_train, y_train)

end_time = time.time()
print(f"RFECV completed in {(end_time - start_time)/60:.2f} minutes")

# Selected features
selected_features = X_train.columns[rfecv.support_]
print("Optimal number of features:", rfecv.n_features_)
print("Selected features:", list(selected_features))

# Plot RFECV performance
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1),
         rfecv.cv_results_['mean_test_score'], marker="o")
plt.xlabel("Number of features selected")
plt.ylabel("Mean CV ROC-AUC")
plt.title("RFECV Feature Selection Curve")
plt.grid(True)
plt.show()



## Step 2 
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score

# reduce X to selected features
Xtr_sel = X_train[selected_features]
Xva_sel = X_valid[selected_features]

models = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000, solver="lbfgs", class_weight="balanced"),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(y_train.value_counts()[0]/y_train.value_counts()[1]),
        eval_metric="aucpr", n_jobs=-1, random_state=42)
}

results = {}

for name, model in models.items():
    start = time.time()
    model.fit(Xtr_sel, y_train)
    proba = model.predict_proba(Xva_sel)[:, 1]
    auc = roc_auc_score(y_valid, proba)
    pr_auc = average_precision_score(y_valid, proba)
    results[name] = {"ROC-AUC": auc, "PR-AUC": pr_auc,
                     "time_min": (time.time() - start)/60}
    print(f"{name}: ROC-AUC={auc:.3f}, PR-AUC={pr_auc:.3f}, Time={results[name]['time_min']:.2f} min")


## Step 3 
##-------------------------------
## Visulaize the model comparison 
##-------------------------------
res_df = pd.DataFrame(results).T
res_df.plot(kind="bar", figsize=(8,5))
plt.title("Model Comparison (Validation Set)")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()


## step 5 : 
##==============================
## Best model & Threshold tuning 
##==============================
#TBA


## Step 6 
##===================================================================
## Decile Analysis 1: Highest - 10 : Lowest , Conversion rate & List 
##==================================================================
# Recreate df_preds if not already available
proba = best_model.predict_proba(Xte_sel)[:, 1]
df_preds = pd.DataFrame({
    "TrueLabel": y_test,
    "Proba": proba
})

# Create decile bins (1 = highest risk, 10 = lowest risk)
df_preds['Decile'] = pd.qcut(
    df_preds['Proba'].rank(method='first', ascending=False),
    10, labels=False
) + 1  # now 1 = top 10% highest scores

# Group by decile and calculate metrics
decile_summary = df_preds.groupby('Decile').agg(
    Count=('TrueLabel', 'count'),
    Conversions=('TrueLabel', 'sum'),
    Avg_Probability=('Proba', 'mean')
).reset_index()

# Calculate conversion rate
decile_summary['ConversionRate'] = (
    decile_summary['Conversions'] / decile_summary['Count']
)

# Calculate lift
overall_rate = df_preds['TrueLabel'].mean()
decile_summary['Lift'] = decile_summary['ConversionRate'] / overall_rate

# Sort so Decile 1 = top risk comes first
decile_summary = decile_summary.sort_values(by='Decile', ascending=True)

# Display
print("=== Decile Analysis Summary (Decile 1 = Highest Risk) ===")
print(decile_summary)


## Step 7 
##=======================================
## Gain Chart / Cumulative Recall Curve 
##======================================

'''
How much of the positive class (HH referrals) you capture as you move down the ranked deciles.
'''

import matplotlib.pyplot as plt

# --- Decile Summary with Cumulative Metrics ---
def decile_gain_chart(y_true, y_proba, n_deciles=10):
    df = pd.DataFrame({"TrueLabel": y_true, "Proba": y_proba})

    # Rank probabilities (highest = top risk)
    df = df.sort_values("Proba", ascending=False).reset_index(drop=True)
    df["Decile"] = pd.qcut(df.index, q=n_deciles, labels=False) + 1

    # Group by decile
    decile_summary = df.groupby("Decile").agg(
        Count=("TrueLabel", "count"),
        Conversions=("TrueLabel", "sum"),
        Avg_Probability=("Proba", "mean")
    ).reset_index()

    # Conversion rate
    decile_summary["ConversionRate"] = (
        decile_summary["Conversions"] / decile_summary["Count"]
    )

    # Cumulative HH (recall curve)
    decile_summary["Cum_Conversions"] = decile_summary["Conversions"].cumsum()
    decile_summary["Cum_Recall"] = (
        decile_summary["Cum_Conversions"] / decile_summary["Conversions"].sum()
    )

    # Lift
    overall_rate = df["TrueLabel"].mean()
    decile_summary["Lift"] = decile_summary["ConversionRate"] / overall_rate

    return decile_summary

# --- Run for your model ---
decile_summary = decile_gain_chart(y_test, proba, n_deciles=10)
print(decile_summary)

# --- Gain Chart (Cumulative Recall) ---
plt.figure(figsize=(8,5))
plt.plot(decile_summary["Decile"], decile_summary["Cum_Recall"],
         marker="o", label="Cumulative Recall")
plt.plot([1,10],[0,1], "--", color="gray", label="Random Model")  # baseline
plt.xlabel("Decile (1 = Highest Risk)")
plt.ylabel("Cumulative Recall (HH captured)")
plt.title("Gain Chart – HH Propensity Model")
plt.legend()
plt.grid(True)
plt.show()



