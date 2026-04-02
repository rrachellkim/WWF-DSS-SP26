import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# df = pd.read_csv("DSS/WWF-DSS-SP26/amazon_model/training_amazon_cleaned.csv")
# df = pd.read_csv("DSS/model_training_cleaned.csv")
# df = pd.read_csv("DSS/new_amazon_model_training_data.csv")
df = pd.read_csv("DSS/new_single_model_training_data.csv",
                 encoding="latin1")

target_col = "Priority Place"
df[target_col] = df[target_col].astype(str).str.strip()
df = df[df[target_col].notna()]

text_cols = [
    "Cost Center",
    "Program Code",
    "Grant",
    "Country",
]

for c in text_cols:
    if c not in df.columns:
        df[c] = ""

df[text_cols] = df[text_cols].astype(str)
df["text_all"] = df[text_cols].fillna("").agg(" ".join, axis=1)

X = df["text_all"]
y = df[target_col]

n_classes = y.nunique()
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y if n_classes > 1 else None,
    random_state=42,
)

rf_clf = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000
    )),
    ("rf", RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample"  
    )),
])

rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)

print("Train accuracy:", rf_clf.score(X_train, y_train))
print("Test accuracy:", rf_clf.score(X_test, y_test))
print()
print("Classification report:")
print(classification_report(y_test, rf_clf.predict(X_test)))

labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
 
fig, ax = plt.subplots(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(
    ax=ax,
    cmap="Blues",
    colorbar=True,
    xticks_rotation=45,
)
 
ax.set_title("Random Forest — Confusion Matrix", fontsize=14, pad=16)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("True", fontsize=11)
ax.xaxis.set_tick_params(labelsize=9)
ax.yaxis.set_tick_params(labelsize=9)
 
plt.tight_layout()
plt.savefig("DSS/confusion_matrix.png", dpi=150, bbox_inches="tight")
print("Saved confusion_matrix.png")
 