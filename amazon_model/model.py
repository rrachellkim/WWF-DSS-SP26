import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("DSS/training_amazon_cleaned.csv")

target_col = "priority_place_simple"

if target_col not in df.columns:
    priority_map = {
        "amazon": "Amazon",
        "arctic": "Arctic",
        "coral triangle": "Coral Triangle",
        "yucatan": "Mesoamerica",
        "swio": "SWIO",
        "eastern pacific": "Eastern Pacific",
        "southern africa": "Southern Africa",
    }
    df["IP or Country"] = df["IP or Country"].astype(str)
    ip_lower = df["IP or Country"].str.lower().fillna("")
    labels = []
    for v in ip_lower:
        label = "Other"
        for k, lab in priority_map.items():
            if k in v:
                label = lab
                break
        labels.append(label)
    df[target_col] = labels

df = df[df[target_col].notna()]

text_cols = [
    "WD Cost Center Hierarchy",
    "Grant",
    "Program",
    "Project Area",
    "IP or Country",
]

for c in text_cols:
    if c not in df.columns:
        df[c] = ""

df[text_cols] = df[text_cols].astype(str)
df["text_all"] = df[text_cols].fillna("").agg(" ".join, axis=1)

X = df["text_all"]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
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

print("Train accuracy:", rf_clf.score(X_train, y_train))
print("Test accuracy:", rf_clf.score(X_test, y_test))
print()
print("Classification report:")
print(classification_report(y_test, rf_clf.predict(X_test)))
