# Product Category Prediction

Ovaj projekat automatski predviđa kategoriju proizvoda na osnovu naziva.  
Koristi **Python**, **scikit-learn** i **Jupyter Notebook** za analizu podataka, treniranje i testiranje modela.

---

## 📂 Struktura projekta

product-category/
│
├── data/
│ └── products.csv # Skup podataka sa proizvodima
│
├── models/
│ └── product_model.pkl # Sačuvani trenirani model
│
├── notebooks/
│ └── 01_EDA_and_Model.ipynb # Notebook sa analizom i treniranjem modela
│
├── src/
│ ├── train_model.py # Skripta za treniranje modela
│ └── predict_category.py # Skripta za predikciju kategorije
│
└── README.md


---

## 🛠️ Zahtevi

- Python 3.9+  
- Biblioteke:

```bash
pip install pandas scikit-learn joblib

Opcionalno za Jupyter:
pip install jupyter

🚀 Pokretanje projekta
1️⃣ Treniranje modela

Pokreni treniranje iz CMD ili terminala:

python src/train_model.py


Ovo učitava data/products.csv, trenira model i čuva ga u models/product_model.pkl.

2️⃣ Predikcija kategorije

Pokreni interaktivnu skriptu:

python src/predict_category.py


Unesi naziv proizvoda kada budeš upitan:

Naziv > iphone 7 32gb gold
Predviđena kategorija: Mobile Phones (verovatnoća: 0.952)


Da izađeš iz aplikacije, upiši exit.