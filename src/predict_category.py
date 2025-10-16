# src/predict_category.py
import re
import joblib
import sys
from pathlib import Path

MODEL_PATH = Path("models/product_model.pkl")

def preprocess_title(title: str) -> str:
    """Osnovno čišćenje teksta - ista logika koju smo koristili pri treniranju."""
    t = str(title).lower()
    t = re.sub(r'[^a-z0-9 ]', ' ', t)   # zameni sve što nije slovo/broj/space sa space
    t = re.sub(r'\s+', ' ', t).strip()  # sredi višestruke space i trime
    return t

def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model nije pronađen na {path}. Pokreni src/train_model.py prvo.")
    return joblib.load(path)

def predict(model, title: str):
    proc = preprocess_title(title)
    # model očekuje originalan pipeline ulaz (string)
    pred = model.predict([proc])[0]
    proba = None
    try:
        if hasattr(model, "predict_proba"):
            proba_vals = model.predict_proba([proc])[0]
            # uzmi najveću verovatnoću i labelu (ako model vraća classes_)
            proba = max(proba_vals)
    except Exception:
        proba = None
    return pred, proba

def interactive_loop(model):
    print("Unesi naziv proizvoda (ili 'exit' za kraj).")
    while True:
        try:
            title = input("Naziv > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nKraj.")
            break
        if not title:
            continue
        if title.lower() in ("exit", "quit"):
            print("Kraj.")
            break
        try:
            cat, p = predict(model, title)
            if p is None:
                print(f"Predviđena kategorija: {cat}\n")
            else:
                print(f"Predviđena kategorija: {cat} (verovatnoća: {p:.3f})\n")
        except Exception as e:
            print("Greška pri predikciji:", e)
            break

if __name__ == "__main__":
    # dozvoli naslov kao argument: python src/predict_category.py "iphone 7 32gb"
    if len(sys.argv) > 1:
        # uzmi sve argumente i spoji u naslov
        title = " ".join(sys.argv[1:])
        model = load_model(MODEL_PATH)
        cat, p = predict(model, title)
        if p is None:
            print(cat)
        else:
            print(f"{cat} ({p:.3f})")
    else:
        model = load_model(MODEL_PATH)
        interactive_loop(model)
