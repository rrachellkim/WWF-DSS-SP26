import pandas as pd
import numpy as np

df = pd.read_csv(
    "DSS/training_amazon.csv",
    encoding="latin1",
)
print(f"Raw shape: {df.shape}")

junk_cols = [c for c in df.columns if c.startswith("Unnamed")] + ["OG Core Distro"]
df.drop(columns=junk_cols, inplace=True)

df.columns = df.columns.str.strip()

df = df[df["Program Totals"].isna()].drop(columns=["Program Totals"])

str_cols = df.select_dtypes(include="str").columns
df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

df.replace("", np.nan, inplace=True)

df["Funding Type"] = df["Funding Type"].replace("UNR ", "UNR")

df["Total"] = (
    df["Total"]
    .str.replace(",", "", regex=False)
    .str.replace(r"[()]+", "-", regex=True)  
    .pipe(pd.to_numeric, errors="coerce")
)

df.rename(columns={"Type": "Type", "IP Country": "IP Country"}, inplace=True)

df.reset_index(drop=True, inplace=True)

print(f"Clean shape: {df.shape}")
print("\nNull counts after cleaning:")
print(df.isnull().sum().to_string())
print("\nDtypes:")
print(df.dtypes.to_string())

df.to_csv("DSS/training_amazon_cleaned.csv", index=False)
print("Data cleaning done")