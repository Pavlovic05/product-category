# src/train_model.py
import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path

# Putanje
DATA_PATH = Path("data/products.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "product_model.pkl"

# 1️⃣ Učitaj podatke
print("Učitavanje podataka...")
df = pd.read_csv(DATA_PATH)

# Očisti nazive kolona od praznih mesta
df.columns = df.columns.str.strip()

# Koristi samo kolone koje su nam potrebne
df = df[["Product Title", "Category Label"]].dropna()

# 2️⃣ Preprocessing (isto kao u predict skripti)
def preprocess_title(title: str) -> str:
    t = str(title).lower()
    t = re.sub(r'[^a-z0-9 ]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

df["clean_title"] = df["Product Title"].apply(preprocess_title)

# 3️⃣ Podela skupa
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_title"], df["Category Label"], test_size=0.2, random_state=42, stratify=df["Category Label"]
)

# 4️⃣ TF-IDF + Logistic Regression pipeline
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000)),
    ("clf", LogisticRegression(max_iter=1000))
])

print("Treniranje modela...")
pipeline.fit(X_train, y_train)

# 5️⃣ Evaluacija
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTačnost modela: {acc:.3f}")
print("\nKlasifikacioni izveštaj:")
print(classification_report(y_test, y_pred))

# 6️⃣ Sačuvaj model
joblib.dump(pipeline, MODEL_PATH)
print(f"\n✅ Model sačuvan u: {MODEL_PATH}")
