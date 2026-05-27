import pandas as pd
import numpy as np

df = pd.read_csv("DSS/new_single_model_training_data.csv", encoding="latin1")

df.columns = df.columns.str.strip()

cost_col = 'Cost Center'
prog_col = 'Program Code'

df[cost_col] = df[cost_col].fillna('').astype(str)
df[prog_col] = df[prog_col].fillna('').astype(str)

combined = (df[cost_col] + ' ' + df[prog_col]).str.lower()

df['Big Bet Tag'] = 'Other'

finance_patterns = [
    'finance',
]

resilience_patterns = [
    'resilience'
]

infrastructure_patterns = [
    'infrastructure'
]

# finance_patterns = [
#     'finance', 'financial', 'accounting', 'budget', 'treasury'
# ]

# resilience_patterns = [
#     'resilience', 'resilient', 'sustainable', 'sustainability',
#     'climate', 'adaptation'
# ]

# infrastructure_patterns = [
#     'infrastructure', 'transport', 'transit', 'roads',
#     'bridges', 'utilities', 'facilities'
# ]

def matches_any(text_series, patterns):
    mask = False
    for pat in patterns:
        mask = mask | text_series.str.contains(pat, na=False)
    return mask

mask_fin = matches_any(combined, finance_patterns)
df.loc[mask_fin, 'Big Bet Tag'] = 'Finance'

mask_res = matches_any(combined, resilience_patterns)
df.loc[mask_res, 'Big Bet Tag'] = 'Resilience'

mask_inf = matches_any(combined, infrastructure_patterns)
df.loc[mask_inf, 'Big Bet Tag'] = 'Infrastructure'

df = pd.read_csv("DSS/model_training_with_bb.csv", encoding="latin1")

labels = ["Finance", "Resilience", "Infrastructure"]

for label in labels:
    true_mask = df["Big Bet"] == label
    n_true = true_mask.sum()
    if n_true == 0:
        print(f"{label}: no rows with this true label")
        continue

    correct = (df["Big Bet Tag"] == label) & true_mask
    n_correct = correct.sum()
    proportion = n_correct / n_true

    print(f"{label}: {n_correct} / {n_true} correctly tagged "
          f"({proportion:.3f} accuracy)")
    
mask_three = df["Big Bet"].isin(labels)
overall_correct = (df["Big Bet"] == df["Big Bet Tag"]) & mask_three
overall_accuracy = overall_correct.sum() / mask_three.sum()
print(f"Overall accuracy on Finance/Resilience/Infrastructure: "
      f"{overall_accuracy:.3f}")

df.to_csv("DSS/model_training_with_bb.csv", index=False)