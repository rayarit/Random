# Step 1 — Imports
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit

import xgboost as xgb
from datetime import datetime, timedelta

# 🔧 Step 2 — Load + Prepare Data
df = pd.read_csv("your_data.csv")

# --- Create time index ---
df['date'] = pd.to_datetime(df['WEEK_END_DATE'])
df = df.sort_values(['CHANNEL', 'WAVE_NUM', 'date'])

# --- Series ID ---
df['series_id'] = df['CHANNEL'].astype(str) + "_" + df['WAVE_NUM'].astype(str)

# 🔧 Step 3 — Feature Engineering
Lag features
LAGS = [1,2,3,4,12]

for lag in LAGS:
    df[f'lag_{lag}'] = (
        df.groupby('series_id')['ENROLLED_SCRIPTS']
        .shift(lag)
    )

# Rolling stats
df['rolling_mean_4'] = (
    df.groupby('series_id')['ENROLLED_SCRIPTS']
    .shift(1)
    .rolling(4)
    .mean()
)

df['rolling_std_4'] = (
    df.groupby('series_id')['ENROLLED_SCRIPTS']
    .shift(1)
    .rolling(4)
    .std()
)

# Time features
df['weekofyear'] = df['date'].dt.isocalendar().week

df['sin_week'] = np.sin(2*np.pi*df['weekofyear']/52)
df['cos_week'] = np.cos(2*np.pi*df['weekofyear']/52)

# Encode categorical
df = pd.get_dummies(df, columns=['CHANNEL'], drop_first=True)

# Drop NA from lags
df_model = df.dropna().reset_index(drop=True)

## 8️⃣ Train / Validation Split
# Last 13 weeks validation
cutoff_date = df_model['date'].max() - pd.Timedelta(weeks=13)

train_df = df_model[df_model['date'] <= cutoff_date]
val_df   = df_model[df_model['date'] > cutoff_date]

## 9#️⃣ Feature List
TARGET = 'ENROLLED_SCRIPTS'

FEATURES = [col for col in df_model.columns if col not in [
    'ENROLLED_SCRIPTS',
    'date',
    'series_id',
    'WEEK_END_DATE'
]]

## 🔟 Model Training (XGBoost)
model = xgb.XGBRegressor(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)

model.fit(
    train_df[FEATURES],
    train_df[TARGET]
)

## 1️⃣1️⃣ Validation Prediction
val_df['pred'] = model.predict(val_df[FEATURES])

## 1#️⃣2️⃣ Accuracy Metrics
rmse = np.sqrt(mean_squared_error(val_df[TARGET], val_df['pred']))
mape = mean_absolute_percentage_error(val_df[TARGET], val_df['pred'])

# Residual STD
residuals = val_df[TARGET] - val_df['pred']
std_dev = np.std(residuals)

rmse_std_ratio = rmse / std_dev

print("RMSE :", rmse)
print("MAPE :", mape)
print("Residual STD :", std_dev)
print("RMSE / STD :", rmse_std_ratio)


## 👉 Interpretation:

## Ratio	Meaning
#< 1	Model very stable
#~1	Acceptable
#>1.5	Poor variance capture
## 1️⃣3️⃣ Hyperparameter Tuning (Optimized)
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    "n_estimators": [400,600,800,1000],
    "max_depth": [4,5,6,8],
    "learning_rate": [0.01,0.03,0.05],
    "subsample": [0.7,0.8,0.9],
    "colsample_bytree": [0.7,0.8,0.9]
}

tscv = TimeSeriesSplit(n_splits=3)

search = RandomizedSearchCV(
    xgb.XGBRegressor(objective='reg:squarederror'),
    param_grid,
    n_iter=25,
    cv=tscv,
    scoring='neg_root_mean_squared_error',
    verbose=1,
    n_jobs=-1
)

search.fit(train_df[FEATURES], train_df[TARGET])

best_model = search.best_estimator_

## 1️⃣4️⃣ 13-Week Forecasting (Recursive)
import numpy as np
import pandas as pd

HORIZON = 13
future_preds = []

last_df = df_model.copy()

# Ensure sorted for correct tail/rolling behavior
last_df = last_df.sort_values(['series_id', 'date']).reset_index(drop=True)

min_date = last_df['date'].min()

for step in range(HORIZON):

    # Next forecast date (global weekly step)
    next_date = last_df['date'].max() + pd.Timedelta(weeks=1)

    # Base row for each series = last known row
    temp = (
        last_df.sort_values(['series_id', 'date'])
              .groupby('series_id', as_index=False)
              .tail(1)
              .copy()
    )

    # Set future date
    temp['date'] = next_date

    # -------------------------
    # Recompute time features (match what you used in training)
    # -------------------------
    temp['weekofyear'] = temp['date'].dt.isocalendar().week.astype(int)
    temp['sin_week'] = np.sin(2 * np.pi * temp['weekofyear'] / 52)
    temp['cos_week'] = np.cos(2 * np.pi * temp['weekofyear'] / 52)

    # Trend index (weekly)
    temp['time_idx'] = ((temp['date'] - min_date).dt.days // 7).astype(int)

    # -------------------------
    # Recompute lag features per series (SAFE alignment)
    # -------------------------
    for lag in LAGS:
        lag_series = (
            last_df.groupby('series_id')['ENROLLED_SCRIPTS']
                   .shift(lag)
        )
        # take the latest lag value per series (align with temp rows)
        temp[f'lag_{lag}'] = (
            pd.concat([last_df[['series_id']], lag_series.rename('v')], axis=1)
              .sort_values(['series_id'])
              .groupby('series_id')
              .tail(1)['v']
              .values
        )

    # -------------------------
    # Recompute rolling stats per series (SAFE alignment)
    # -------------------------
    roll_mean_4 = (
        last_df.groupby('series_id')['ENROLLED_SCRIPTS']
               .rolling(4)
               .mean()
               .reset_index(level=0, drop=True)
    )

    temp['rolling_mean_4'] = (
        pd.concat([last_df[['series_id']], roll_mean_4.rename('v')], axis=1)
          .sort_values(['series_id'])
          .groupby('series_id')
          .tail(1)['v']
          .values
    )

    # If you also trained with rolling_std_4, compute it too
    if 'rolling_std_4' in FEATURES or 'rolling_std_4' in temp.columns:
        roll_std_4 = (
            last_df.groupby('series_id')['ENROLLED_SCRIPTS']
                   .rolling(4)
                   .std()
                   .reset_index(level=0, drop=True)
        )
        temp['rolling_std_4'] = (
            pd.concat([last_df[['series_id']], roll_std_4.rename('v')], axis=1)
              .sort_values(['series_id'])
              .groupby('series_id')
              .tail(1)['v']
              .values
        )

    # -------------------------
    # IMPORTANT:
    # If you used exogenous variables that are UNKNOWN in the future
    # (e.g., OUTREACHED_SCRIPTS), you must decide how to fill them.
    # Option A (simple): keep last observed value per series
    # Option B: set to 0 / planned campaign values
    # -------------------------
    if 'OUTREACHED_SCRIPTS' in temp.columns and 'OUTREACHED_SCRIPTS' in FEATURES:
        # Keep last known outreach per series (baseline assumption)
        temp['OUTREACHED_SCRIPTS'] = (
            last_df.sort_values(['series_id', 'date'])
                  .groupby('series_id')['OUTREACHED_SCRIPTS']
                  .tail(1)
                  .values
        )

    # -------------------------
    # Predict next step
    # -------------------------
    # Ensure all FEATURES exist in temp
    missing = [c for c in FEATURES if c not in temp.columns]
    if missing:
        raise ValueError(f"Missing required features in temp: {missing}")

    temp['pred'] = model.predict(temp[FEATURES])

    # Feed prediction back as ENROLLED_SCRIPTS for next step lags
    temp['ENROLLED_SCRIPTS'] = temp['pred']

    # Save and append to last_df for next iteration
    future_preds.append(temp[['series_id', 'date', 'pred']].copy())

    last_df = pd.concat([last_df, temp], ignore_index=True)
    last_df = last_df.sort_values(['series_id', 'date']).reset_index(drop=True)

# Final 13-week forecast dataframe
forecast_13w = pd.concat(future_preds, ignore_index=True)

forecast_13w.head()
