# Random

🔹 KS Test Insights

Visit counts (PCP, AWV, APPT) → High KS drift is expected. Filling NaN with 0 creates a spike at zero, which reflects real members with no visits (not bad drift).

Monetary fields (COPAY, CHARGE) → Moderate drift, but again valid since 0 means no claims.

Other fields (COINS, SPC, ALLOW, flags) → Stable, distributions unchanged.

✅ Takeaway: KS drift here reflects true non-utilization, not imputation error. Using 0 + _MISSING flags preserves both signal and interpretability.
