from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Načteme Iris dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Omezíme počet vzorků a vláken (bez multithreadingu)
X_train = X_train[:30]
y_train = y_train[:30]

# Trénink modelu
print("Začínám trénink...")

model = XGBClassifier(
    use_label_encoder=False,
    eval_metric="mlogloss",
    n_jobs=1,             # ⚠️ OMEZÍ THREADING
    verbosity=2           # ⚠️ ZOBRAZENÍ DEBUG INFO
)

model.fit(X_train, y_train)

print("✅ Trénink dokončen.")
