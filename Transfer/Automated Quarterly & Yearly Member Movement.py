from pyspark.sql import functions as F
import re
import os

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
LOB = "medicaid"                 # "medicaid", "commercial", or "medicare"
YEARS = [2023, 2024]             # list of years to analyze
BASE_PATH = adls_results_path + f"/lvl1/feature-store/{LOB}/group-a-train/"
RX_FOLDER = "base/rxclms/"
ID_COL = "sdr_person_id"
# ---------------------------------------------------------------------------


def list_snapshot_folders(base_path):
    """
    Lists all available snapshot folders like 20220131, 20220430, etc.
    (depends on Databricks/ADLS API — here assuming dbutils.fs.ls)
    """
    folders = [f.name.replace("/", "") for f in dbutils.fs.ls(base_path)]
    valid = [f for f in folders if re.match(r"\d{8}", f)]
    return sorted(valid)


def load_snapshot(base_path, snapshot):
    """
    Loads a parquet file for a given snapshot date.
    """
    path = f"{base_path}{snapshot}/{RX_FOLDER}"
    try:
        df = spark.read.parquet(path).select(ID_COL).distinct()
        return df
    except Exception as e:
        print(f"⚠️ Skipping {snapshot} — {str(e)}")
        return None


def member_movement(prev_df, curr_df, period_name):
    """
    Calculates retained, new, and left members between two snapshots.
    """
    if prev_df is None or curr_df is None:
        return (period_name, 0, 0, 0)
    retained = prev_df.join(curr_df, ID_COL, "inner").count()
    left = prev_df.join(curr_df, ID_COL, "left_anti").count()
    new = curr_df.join(prev_df, ID_COL, "left_anti").count()
    return (period_name, retained, new, left)


def quarterly_and_yearly_analysis(base_path, years):
    """
    Runs the quarterly + yearly movement analysis automatically for given years.
    """
    folders = list_snapshot_folders(base_path)
    # filter only selected years
    selected = [f for f in folders if int(f[:4]) in years]
    selected = sorted(selected)

    if not selected:
        print("⚠️ No matching snapshots found for given years.")
        return None

    results = []
    prev_df = None
    prev_snapshot = None

    # # QUARTERLY COMPARISON
    # for snapshot in selected:
    #     curr_df = load_snapshot(base_path, snapshot)
    #     if prev_df is not None:
    #         period = f"{prev_snapshot[:6]}→{snapshot[:6]}"
    #         results.append(member_movement(prev_df, curr_df, period))
    #     prev_df, prev_snapshot = curr_df, snapshot

    # # Convert to Spark DataFrame
    # quarterly_df = spark.createDataFrame(results, ["Period", "Retained", "New_Members", "Left_Members"])

    # QUARTERLY COMPARISON (every 3rd snapshot)
    quarterly_results = []
    for i in range(0, len(selected) - 3, 3):   # step by 3
        snap1 = selected[i]
        snap2 = selected[i + 3]
        df1 = load_snapshot(base_path, snap1)
        df2 = load_snapshot(base_path, snap2)
        period = f"{snap1[:6]}→{snap2[:6]}"
        quarterly_results.append(member_movement(df1, df2, period))
    
    quarterly_df = spark.createDataFrame(quarterly_results, ["Period", "Retained", "New_Members", "Left_Members"])


    # YEARLY COMPARISON (if multiple years)
    yearly_results = []
    year_groups = sorted(set([f[:4] for f in selected]))

    if len(year_groups) >= 2:
        for i in range(len(year_groups) - 1):
            y1, y2 = year_groups[i], year_groups[i + 1]
            # pick first snapshot of each year for comparison
            snap1 = [s for s in selected if s.startswith(y1)][0]
            snap2 = [s for s in selected if s.startswith(y2)][0]
            df1, df2 = load_snapshot(base_path, snap1), load_snapshot(base_path, snap2)
            yearly_results.append(member_movement(df1, df2, f"{y1}→{y2}"))

    yearly_df = spark.createDataFrame(yearly_results, ["Period", "Retained", "New_Members", "Left_Members"]) if yearly_results else None

    return quarterly_df, yearly_df


from pyspark.sql import functions as F

def add_movement_metrics(df):
    """
    Adds retention, churn, new member rate, and net growth percentage metrics.
    """
    df = (
        df.withColumn("Total_Members_Current", F.col("Retained") + F.col("New_Members"))
          .withColumn("Total_Members_Previous", F.col("Retained") + F.col("Left_Members"))
          .withColumn("Retention_Rate", (F.col("Retained") / F.col("Total_Members_Previous") * 100))
          .withColumn("Churn_Rate", (F.col("Left_Members") / F.col("Total_Members_Previous") * 100))
          .withColumn("New_Member_Rate", (F.col("New_Members") / F.col("Total_Members_Current") * 100))
          .withColumn("Net_Growth_Rate", ((F.col("Total_Members_Current") - F.col("Total_Members_Previous")) / F.col("Total_Members_Previous") * 100))
    )
    return df



# ---------------------------------------------------------------------------
# RUN THE ANALYSIS
# ---------------------------------------------------------------------------
quarterly_df, yearly_df = quarterly_and_yearly_analysis(BASE_PATH, YEARS)

if quarterly_df:
    print(f"📊 Quarterly Member Movement for {LOB.upper()}")
    display(quarterly_df)

if yearly_df:
    print(f"📅 Yearly Member Movement for {LOB.upper()}")
    display(yearly_df)
