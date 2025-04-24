## 🔎 Skewness & Kurtosis Check: `credit_bal_studentloan_60dpd`

```python
from pyspark.sql.functions import skewness, kurtosis

df_main.select(
    skewness("credit_bal_studentloan_60dpd").alias("skewness"),
    kurtosis("credit_bal_studentloan_60dpd").alias("kurtosis")
).show()
