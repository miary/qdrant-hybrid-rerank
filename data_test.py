import pandas as pd

df = pd.read_parquet("hf://datasets/truehealth/medicationqa/data/train-00000-of-00001-7427a10e891759be.parquet")
print(df.describe())
print(df.columns)
