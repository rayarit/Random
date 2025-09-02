# Random

Structural Utilization & Cost Fields (e.g., TOTAL_COPAY_AMT, TOTAL_COINS_AMT, TOTAL_CHARGE_AMT, TOTAL_ALLOW_AMT, visit counts)

Reason: These are annual sums/counts.

NaN means no claims/visits recorded → structurally equivalent to 0 utilization.

Action: Fill with 0.0 and add a _MISSING flag to preserve information about whether data was absent vs truly zero.

Binary Flags (e.g., INCURRED_DURING_STAY_FLAG)

Reason: Null indicates no event recorded → safest assumption is 0 (did not occur).

Action: Fill with 0 and add a _MISSING flag to capture uncertainty in reporting.

Low-Null Monetary/Unit Fields (e.g., TOTAL_NET_AMT, TOTAL_UNITS_CNT)

Reason: Null rate <1%, so missingness is not structural but rare data gaps.

Action: Fill with median (robust to skew) to avoid biasing distribution, optional _MISSING flag for audit.

Validation Step (KS Test)

Reason: Ensure that imputation does not distort feature distributions.

Action: Run KS test before vs after imputation; flag features with KS > 0.1 for review.
