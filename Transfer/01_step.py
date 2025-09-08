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

