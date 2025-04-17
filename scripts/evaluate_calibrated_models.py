import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, log_loss
from utils.data_loader import load_data_by_league
from utils.feature_engineering_match_result import generate_match_result_features


def apply_isotonic_calibration(probs, isotonic_models):
    calibrated_probs = np.zeros_like(probs)
    for i in range(probs.shape[1]):
        calibrated_probs[:, i] = isotonic_models[f"class_{i}"].predict(probs[:, i])

    # Prevence NaN – pokud je součet 0, nastavíme na 1, aby nedošlo k dělení nulou
    row_sums = calibrated_probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    calibrated_probs = calibrated_probs / row_sums
    return calibrated_probs


def evaluate_predictions(y_true, y_pred, probs, title):
    print(f"\n📊 Výsledky – {title}")
    print(classification_report(y_true, y_pred, target_names=["Výhra domácích", "Remíza", "Výhra hostů"]))
    print("Maticová chyba:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    ConfusionMatrixDisplay(cm, display_labels=["H", "D", "A"]).plot(cmap="Blues")
    plt.title(f"Confusion Matrix – {title}")
    plt.show()

    # ROC AUC a Log Loss
    try:
        roc_auc = roc_auc_score(pd.get_dummies(y_true), probs, multi_class="ovr")
        print(f"ROC AUC: {roc_auc:.4f}")
    except Exception as e:
        print(f"ROC AUC nelze spočítat: {e}")

    try:
        loss = log_loss(y_true, probs)
        print(f"Log loss: {loss:.4f}")
    except Exception as e:
        print(f"Log loss nelze spočítat: {e}")


def plot_feature_importance(model, feature_names, title):
    importances = model.feature_importances_
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    importance_df = importance_df.sort_values("Importance", ascending=False).head(20)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x="Importance", y="Feature", palette="viridis")
    plt.title(f"Top 20 Feature Importances – {title}")
    plt.tight_layout()
    plt.show()


def evaluate_calibrated_models(league_code):
    print(f"\nEvaluace kalibrovaných modelů pro ligu {league_code}")
    df = load_data_by_league(league_code)
    df_test = df.iloc[-int(len(df)*0.2):]
    df_test_fe = generate_match_result_features(df_test, mode="train")

    X_test = df_test_fe.drop(columns=["HomeTeam", "AwayTeam", "Date", "target_result"])
    y_test = df_test_fe["target_result"]

    model = joblib.load(f"models/{league_code}_result_model.joblib")
    platt_model = joblib.load(f"models/{league_code}_result_model_platt.joblib")
    isotonic_models = joblib.load(f"models/{league_code}_result_model_isotonic.joblib")

    feature_names = X_test.columns.tolist()
    probs = model.predict_proba(X_test)

    # Nekalibrovaný model
    y_pred_raw = model.predict(X_test)
    evaluate_predictions(y_test, y_pred_raw, probs, "XGBoost (nekalibrovaný)")

    # Platt
    platt_preds = platt_model.predict(probs)
    platt_probs = platt_model.predict_proba(probs)
    evaluate_predictions(y_test, platt_preds, platt_probs, "Platt Scaling")

    # Isotonic
    probs_isotonic = apply_isotonic_calibration(probs, isotonic_models)
    isotonic_preds = np.argmax(probs_isotonic, axis=1)
    evaluate_predictions(y_test, isotonic_preds, probs_isotonic, "Isotonic Regression")

    # Feature importance
    plot_feature_importance(model, feature_names, league_code)


if __name__ == "__main__":
    evaluate_calibrated_models("E0")
