# 🔷 PySpark VectorAssembler + Feature Metadata + Broadcast 

## 📌 Objective
To prepare feature vectors for ML models in PySpark using `VectorAssembler`, and extract + broadcast metadata for feature interpretability and efficient distributed processing.

---

## 🧩 Step 1: Vector Assembler to Create a Feature Vector

```python
from pyspark.ml.feature import VectorAssembler, StringIndexer

vecassm = VectorAssembler(outputCol='features')
vecassm.setInputCols(input_cols)  # input_cols = list of feature column names

df = vecassm.transform(df)  # Output: Adds a new 'features' column with all features in vector form

```
### ✅ Why ? --> Combine multiple columns into  a single feature vector (tabular data to ML ready vectorized format) 

## 🧩 Step 2: Extract Feature Index to Name Mapping

```python 

import pandas as pd

pandasDF = pd.DataFrame(
    df.schema["features"].metadata["ml_attr"]["attrs"]["binary"] +
    df.schema["features"].metadata["ml_attr"]["attrs"]["numeric"]
).sort_values("idx")

feature_dict = dict(zip(pandasDF["idx"], pandasDF["name"]))

```

### ✅ Why ? --> After assembling, feature names are embedded in metadata.

## 🧩 Step 3: Broadcast the Feature Mapping to All Nodes
``` python
feature_dict_broad = sc.broadcast(feature_dict)
```
### ✅ Why? --> Efficient distribution of the dictionary to all Spark workers.


