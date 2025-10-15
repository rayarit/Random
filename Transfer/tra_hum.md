2.2 Repository Structure & Configuration

The MLPModelTemplate contains production-ready folders for training, scoring, configuration, and deployment. Only specific sections are editable by data scientists (e.g., src/train, src/score).
It is important that any additions or changes still follow the deployment template format, as this format is mandatory for production and testing.
You may also create a new Databricks cluster, but it must follow the approved environment standards.

2.3 Data Exploration, Feature Building & Training

Exploratory data analysis and feature engineering should be performed within Databricks using approved data sources.
The template repo provides training templates that can be reused or adapted instead of writing ad-hoc notebooks.
This helps maintain consistency and makes the model easier to integrate into the deployment pipeline.

2.4 ADLS Directory Structure (Production Alignment)

As datasets and artifacts are generated, the model’s directory in Azure Data Lake Storage (ADLS) must follow the Model ADLS Structure standards.
All intermediate and processed data should be stored in the correct folders to support traceability, reproducibility, and production handoff.
Development must use production-compliant Databricks runtimes to avoid compatibility issues during deployment.

2.5 Naming Conventions (Mandatory)

Consistent naming is required across:

Repos

ADLS folders

Notebooks and pipelines

MLflow model objects

Standard names enable automation, easier collaboration, and seamless deployment into FlorenceAI.

2.6 Model Registration and Tracking

All models must be registered in MLflow, including versioning, metadata, and artifacts.
Model objects should be tracked and managed according to the Handling Model Objects guidelines to support reuse, promotion to higher environments, and collaboration with AI Engineering.

##=========
%%spark
import org.apache.spark.sql.functions._

// Define table names
val sourceTable     = "cwp_2025Q3_prediction_16_july_1"
val filterTable     = "testing_sdr_only"
val outputTableName = "filtered_prediction_result"

// Read both tables
val dfMain   = spark.read.synapsesql("dedicatedp1.dbo." + sourceTable)
val dfFilter = spark.read.synapsesql("dedicatedp1.dbo." + filterTable)

// Filter: Exclude SDR_PERSON_IDs found in dfFilter
val dfFiltered = dfMain.join(dfFilter, Seq("SDR_PERSON_ID"), "left_anti")

// Save as a new table
dfFiltered.write
  .mode("overwrite")
  .saveAsTable("dbo." + outputTableName)

// Confirm count
println(s"Filtered row count: ${dfFiltered.count()}")


##==================================================
CREATE TABLE your_table_name (
    sdr_person_id NVARCHAR2(4000),
    propensity_score REAL,
    decile INTEGER,
    Group_label NVARCHAR2(4000),
    MBR_PERS_GEN_KEY NVARCHAR2(4000),
    IDCARD_MBR_ID NVARCHAR2(4000),
    START_ZIP_CD NVARCHAR2(4000),
    CWHH_CENTER NVARCHAR2(4000),
    H_NBR NVARCHAR2(4000),
    END_ZIP_CD NVARCHAR2(4000),
    DISTANCE FLOAT,
    MEMBERS_WITHIN_25_MILES NVARCHAR2(4000),
    MEMBERS_WITHIN_10_MILES NVARCHAR2(4000),
    MEMBERS_WITHIN_50_MILES NVARCHAR2(4000),
    prediction FLOAT
);

GRANT ALL ON your_table_name TO TEAMS;
##========================================
# --- Configure your connection parameters ---
connection_parameters = {
    'account': '<YOUR_ACCOUNT>',
    'user': '<YOUR_USER>',
    'password': '<YOUR_PASSWORD>',
    'role': '<YOUR_ROLE>',
    'warehouse': '<YOUR_WAREHOUSE>',
    'database': '<YOUR_DATABASE>',
    'schema': '<YOUR_SCHEMA>'
}
session = Session.builder.configs(connection_parameters).create()

# --- Target table for analysis ---
TABLE_NAME = "<YOUR_TABLE>"

# --- Load data with Snowpark DataFrame ---
df = session.table(TABLE_NAME)

print(f"Schema for {TABLE_NAME}:")
print(df.schema)

# --- Basic Row and Column Overview ---
row_count = df.count()
col_list = [field.name for field in df.schema.fields]
print(f"Rows: {row_count}, Columns: {len(col_list)}")

# --- Data Type Breakdown ---
type_map = {f.name: f.datatype for f in df.schema.fields}
numeric_cols = [c for c, t in type_map.items() if str(t) in ['FloatType', 'DoubleType', 'DecimalType', 'IntegerType', 'LongType']]
categorical_cols = [c for c, t in type_map.items() if str(t) in ['StringType', 'VarcharType']]

print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

# --- Edge Case Checks: Nulls & Constants ---
for col_name in col_list:
    n_nulls = df.filter(col(col_name).is_null()).count()
    n_distinct = df.select(col_name).distinct().count()
    n_total = row_count

    if n_nulls > 0:
        print(f"Column {col_name}: {n_nulls} nulls ({n_nulls * 100.0 / n_total:.2f}%)")
    if n_distinct == 1:
        print(f"Column {col_name}: Only 1 unique value—consider dropping.")

# --- Duplicate Rows ---
dup_count = df.group_by(col_list).count().filter(col("count") > 1).count()
print(f"Duplicate rows: {dup_count}")

# --- Cardinality Checks (Categoricals) ---
for c in categorical_cols:
    n_unique = df.select(c).distinct().count()
    print(f"Categorical column {c}: {n_unique} unique values.")

# --- Descriptive Stats for Numerics ---
for c in numeric_cols:
    stats = df.agg([
        col(c).min().alias("min"),
        col(c).max().alias("max"),
        col(c).mean().alias("mean"),
        col(c).stddev().alias("stddev"),
        col(c).skew().alias("skew")
    ]).to_pandas()
    print(f"Stats for {c}:\n{stats}")

# --- Outlier/Extreme Value Detection ---
for c in numeric_cols:
    q1 = df.stat.approx_quantile(c, [0.25])[0]
    q3 = df.stat.approx_quantile(c, [0.75])[0]
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    n_outliers = df.filter((col(c) < lower_bound) | (col(c) > upper_bound)).count()
    print(f"Column {c}: Outliers (using 1.5*IQR): {n_outliers}")

# --- Imbalance Checks (Target Column Example) ---
TARGET_COL = "<OPTIONAL_TARGET_COL>"
if TARGET_COL in categorical_cols:
    balance = df.group_by(TARGET_COL).count().to_pandas()
    print(f"Class balance for {TARGET_COL}:\n{balance}")

session.close()


##============ Prompt ===================
Act as a senior data scientist using Snowpark Python for Snowflake.
Objective: Write a single, reusable Python script that performs an automated, robust exploratory data analysis (EDA) and comprehensive data quality assessment for any new table in Snowflake using Snowpark DataFrame API. 
Requirements:The script must work with any table (parameterized table name).
Include connections and credentials placeholders.
Automatically inspect schema and print data types, row/column counts.
Identify and report:
Columns with missing/null/blank values and % missing.
Columns with only one unique value (“constant” columns).
Duplicate rows and count.
Cardinality and uniqueness for all categorical columns.
Descriptive statistics (min, max, mean, stddev, skew) for all numeric columns.
Statistical outliers in numeric columns using both IQR and z-score methods.
Imbalanced distribution for categorical/target columns.
Any columns rarely or never populated.
Data type mismatches or unexpected data types.
Each section must be well-commented and modular so analysts can extend or reuse code blocks.
Output all results in a human-readable format (console print or structured logging).
Edge Cases to Handle:
Columns with all values null or a single repeated value.
Mixed-type columns (e.g., numeric columns containing text).
Unexpected data types or schema drift.
Best Practices:
Close the Snowpark session at the end.
Suggest points of extension (e.g., drift detection, anomaly detection, ML model prepping).
Target Audience: Analytics engineers, data scientists, ML engineers onboarding new Snowflake tables.
Do not hardcode column names; infer all logic dynamically.
