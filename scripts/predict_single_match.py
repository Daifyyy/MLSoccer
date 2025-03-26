import os
import joblib
import numpy as np

# Dynamická cesta k modelu
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "over25_model.joblib")

# Načtení modelu
model = joblib.load(MODEL_PATH)

# ---- UŽIVATELSKÝ VSTUP (může být nahrazen funkcí/GUI) ----
print("Zadej statistiky zápasu:")
HS = int(input("Střely domácího týmu (HS): "))
AS = int(input("Střely hostujícího týmu (AS): "))
HST = int(input("Střely na branku domácího týmu (HST): "))
AST = int(input("Střely na branku hostujícího týmu (AST): "))
HC = int(input("Rohy domácího týmu (HC): "))
AC = int(input("Rohy hostujícího týmu (AC): "))

# ---- Vytvoření vstupního pole ----
match_data = np.array([[HS, AS, HST, AST, HC, AC]])

# ---- Predikce ----
probability = model.predict_proba(match_data)[0][1]
result = model.predict(match_data)[0]

# ---- Výstup ----
print("\nPravděpodobnost, že padne více než 2.5 gólu:")
print(f"{probability * 100:.2f}%")

if result == 1:
    print("✅ Model predikuje: OVER 2.5 goals")
else:
    print("❌ Model predikuje: UNDER 2.5 goals")