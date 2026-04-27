import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)
from matplotlib.patches import Patch

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

PALETTE = {
    "Finance":        "#4C72B0",
    "Resilience":     "#DD8452",
    "Infrastructure": "#55A868",
    "Other":          "#C44E52",
}


df = pd.read_csv("DSS/new_first_model_training_data.csv", encoding="latin1")
print(f"Raw rows: {len(df)}")

def parse_budget(v):
    s = str(v).replace(",", "").strip()
    if s.startswith("(") and s.endswith(")"):
        return -float(s[1:-1])
    try:
        return float(s)
    except ValueError:
        return np.nan

df["Budget"] = df["Budget"].apply(parse_budget)

KEEP = {"Finance", "Resilience", "Infrastructure"}
df["Big Bet"] = df["Big Bet"].str.strip()
df = df.dropna(subset=["Big Bet"]).copy()
df["target"] = df["Big Bet"].apply(lambda x: x if x in KEEP else "Other")

for col in ["Grant", "Cost Center", "Program Code"]:
    df[col] = df[col].fillna("Unknown")


df["log_budget"] = np.sign(df["Budget"].fillna(0)) * np.log1p(np.abs(df["Budget"].fillna(0)))
df["cc_num"]     = df["Cost Center"].str.extract(r"CC(\d+)").astype(float).fillna(-1)
df["grant_id"]   = df["Grant"].str.extract(r"GR0*(\d+)\.").astype(float).fillna(-1)
df["budget_decile"] = pd.qcut(
    df["Budget"].abs().clip(lower=1), q=10, labels=False, duplicates="drop"
).fillna(0).astype(float)

KEYWORDS = ["Unrestricted", "Restricted", "Panorama", "Uncovered",
            "Institutional", "Coverage", "carryforward",
            "Finance", "Resilience", "Infrastructure"]
for kw in KEYWORDS:
    df[f"has_{kw.lower()}"] = df["Grant"].str.contains(kw, case=False, na=False).astype(float)

NUM_COLS = (
    ["log_budget", "cc_num", "grant_id", "budget_decile"] +
    [f"has_{kw.lower()}" for kw in KEYWORDS]
)

tfidf_grant = TfidfVectorizer(max_features=600, ngram_range=(1, 3), sublinear_tf=True, min_df=1)
tfidf_cc    = TfidfVectorizer(max_features=200, ngram_range=(1, 2), sublinear_tf=True, min_df=1)
tfidf_pc    = TfidfVectorizer(max_features=200, ngram_range=(1, 2), sublinear_tf=True, min_df=1)

Xg = tfidf_grant.fit_transform(df["Grant"])
Xc = tfidf_cc.fit_transform(df["Cost Center"])
Xp = tfidf_pc.fit_transform(df["Program Code"])
Xn = sp.csr_matrix(df[NUM_COLS].values)

X = sp.hstack([Xg, Xc, Xp, Xn])
le = LabelEncoder()
y  = le.fit_transform(df["target"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")


CONFIGS = [
    dict(n_estimators=300,  learning_rate=0.10, max_depth=4, subsample=0.9,  min_samples_leaf=1),
    dict(n_estimators=400,  learning_rate=0.08, max_depth=5, subsample=0.85, min_samples_leaf=1),
    dict(n_estimators=500,  learning_rate=0.05, max_depth=5, subsample=0.8,  min_samples_leaf=2),
    dict(n_estimators=600,  learning_rate=0.05, max_depth=6, subsample=0.8,  min_samples_leaf=1),
    dict(n_estimators=800,  learning_rate=0.03, max_depth=6, subsample=0.75, min_samples_leaf=1),
    dict(n_estimators=1000, learning_rate=0.02, max_depth=6, subsample=0.75, min_samples_leaf=1),
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
sweep_results = []

for cfg in CONFIGS:
    model  = GradientBoostingClassifier(**cfg, random_state=42)
    cv_f1  = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1).mean()
    model.fit(X_train, y_train)
    tr_acc = accuracy_score(y_train, model.predict(X_train))
    te_acc = accuracy_score(y_test,  model.predict(X_test))
    te_f1  = f1_score(y_test, model.predict(X_test), average="macro")
    sweep_results.append({**cfg, "cv_f1": cv_f1, "train_acc": tr_acc,
                           "test_acc": te_acc, "test_f1": te_f1, "model": model})
    print(f"  n={cfg['n_estimators']:4d} lr={cfg['learning_rate']:.2f} depth={cfg['max_depth']}"
          f"  |  CV F1={cv_f1:.3f}  train_acc={tr_acc:.3f}  test_acc={te_acc:.3f}")

best       = max(sweep_results, key=lambda r: r["cv_f1"])
best_model = best["model"]
print(f"\n  Best: n={best['n_estimators']}  lr={best['learning_rate']}  depth={best['max_depth']}")


y_train_pred = best_model.predict(X_train)
y_test_pred  = best_model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc  = accuracy_score(y_test,  y_test_pred)
train_f1  = f1_score(y_train, y_train_pred, average="macro")
test_f1   = f1_score(y_test,  y_test_pred,  average="macro")

print(f"Train accuracy: {train_acc*100:.1f}%")
print(f"Test accuracy: {test_acc*100:.1f}%")
print(f"Train F1-macro: {train_f1:.3f}")
print(f"Test F1-macro: {test_f1:.3f}")
print()
print(classification_report(y_test, y_test_pred, target_names=le.classes_))


df_sweep = pd.DataFrame(sweep_results)
labels   = [f"n={r['n_estimators']}\nlr={r['learning_rate']}" for _, r in df_sweep.iterrows()]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Gradient Boosting — Hyperparameter Sweep", fontsize=14, fontweight="bold")

axes[0].plot(labels, df_sweep["cv_f1"],    marker="o", color="#4C72B0", linewidth=2)
axes[0].set_title("CV F1-macro"); axes[0].set_ylabel("Score")
axes[0].tick_params(axis="x", rotation=30); axes[0].set_ylim(0, 1)

axes[1].plot(labels, df_sweep["train_acc"], marker="s", color="#55A868", linewidth=2, label="Train")
axes[1].plot(labels, df_sweep["test_acc"],  marker="^", color="#C44E52", linewidth=2, label="Test")
axes[1].set_title("Accuracy"); axes[1].set_ylabel("Accuracy")
axes[1].legend(); axes[1].tick_params(axis="x", rotation=30); axes[1].set_ylim(0, 1)

axes[2].plot(labels, df_sweep["test_f1"], marker="D", color="#DD8452", linewidth=2)
axes[2].set_title("Test F1-macro"); axes[2].set_ylabel("Score")
axes[2].tick_params(axis="x", rotation=30); axes[2].set_ylim(0, 1)

plt.tight_layout()
plt.savefig("DSS/bb/sweep_results.png", dpi=150, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Best Model  |  Train {train_acc*100:.1f}%  |  Test {test_acc*100:.1f}%",
             fontsize=13, fontweight="bold")

cm   = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Confusion Matrix (Test)", fontweight="bold")
axes[0].tick_params(axis="x", rotation=20)

report  = classification_report(y_test, y_test_pred, target_names=le.classes_, output_dict=True)
f1_vals = {cls: report[cls]["f1-score"] for cls in le.classes_}
colors  = [PALETTE[c] for c in f1_vals.keys()]
bars    = axes[1].bar(f1_vals.keys(), f1_vals.values(), color=colors, edgecolor="white")
axes[1].bar_label(bars, fmt="%.2f", padding=3, fontweight="bold")
axes[1].set_ylim(0, 1.15)
axes[1].axhline(0.8, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="80% line")
axes[1].set_title("Per-Class F1 (Test)", fontweight="bold")
axes[1].set_xlabel("Big Bet Class"); axes[1].set_ylabel("F1 Score")
axes[1].legend(); axes[1].tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig("DSS/bb/model_results.png", dpi=150, bbox_inches="tight")
plt.close()

feat_names = (
    [f"grant: {t}" for t in tfidf_grant.get_feature_names_out()] +
    [f"cc: {t}"    for t in tfidf_cc.get_feature_names_out()]    +
    [f"pc: {t}"    for t in tfidf_pc.get_feature_names_out()]    +
    NUM_COLS
)
fi_df = (pd.DataFrame({"feature": feat_names,
                        "importance": best_model.feature_importances_})
           .sort_values("importance", ascending=False).head(30))

fig, ax = plt.subplots(figsize=(12, 8))
cmap = {"grant": "#4C72B0", "cc": "#55A868", "pc": "#DD8452"}
bar_colors = [
    "#E06C75" if f in NUM_COLS
    else cmap.get(f.split(":")[0].strip(), "#888")
    for f in fi_df["feature"]
]
ax.barh(fi_df["feature"], fi_df["importance"], color=bar_colors, edgecolor="white")
ax.invert_yaxis()
ax.set_title("Top 30 Feature Importances", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance")
ax.legend(handles=[
    Patch(color="#4C72B0", label="Grant TF-IDF"),
    Patch(color="#55A868", label="Cost Center TF-IDF"),
    Patch(color="#DD8452", label="Program Code TF-IDF"),
    Patch(color="#E06C75", label="Numeric / Flags"),
], loc="lower right")
plt.tight_layout()
plt.savefig("DSS/bb/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Data Overview", fontsize=13, fontweight="bold")

counts = df["target"].value_counts()
bars = axes[0].bar(counts.index, counts.values,
                   color=[PALETTE[c] for c in counts.index], edgecolor="white")
axes[0].bar_label(bars, padding=3, fontweight="bold")
axes[0].set_title("Class Distribution"); axes[0].set_xlabel("Class"); axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=20)

sns.boxplot(data=df, x="target", y="Budget", order=list(PALETTE.keys()),
            palette=PALETTE, ax=axes[1], showfliers=False, linewidth=1.2)
axes[1].set_title("Budget by Class (outliers hidden)")
axes[1].set_xlabel("Class")
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
axes[1].tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig("DSS/bb/eda.png", dpi=150, bbox_inches="tight")
plt.close()


print("\n" + "=" * 58)
print("Summary")
print(f"Train accuracy: {train_acc*100:.1f}%")
print(f"Test  accuracy: {test_acc*100:.1f}%")
print(f"Test  F1-macro: {test_f1:.3f}")