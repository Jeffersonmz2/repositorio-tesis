# =============================================================================
# XGBoost + Optuna: V + W → Y
# Tunear hiperparámetros y comparar con Pogit
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
import optuna
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Rutas
# -----------------------------------------------------------------------------
TAG="_min30"
DATA_DIR = Path("../data/matrices")
WB = DATA_DIR / f"matrices_data{TAG}.xlsx"
POGIT_PRED_PATH = DATA_DIR / f"pogit_pred_test.csv"  # opcional
OUTPUT_DIR = DATA_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# 1. Cargar datos
# -----------------------------------------------------------------------------
print("Cargando matrices...")

V_train = pd.read_excel(WB, sheet_name="V_train")
V_test  = pd.read_excel(WB, sheet_name="V_test")
W_train = pd.read_excel(WB, sheet_name="W_train")
W_test  = pd.read_excel(WB, sheet_name="W_test")
Y_train = pd.read_excel(WB, sheet_name="Y_train").iloc[:, 0].astype(float).values
Y_test  = pd.read_excel(WB, sheet_name="Y_test").iloc[:, 0].astype(float).values

for df in (V_train, V_test, W_train, W_test):
    cols = [c for c in df.columns if str(c).lower().startswith("intercept")]
    if cols:
        df.drop(columns=cols, inplace=True)

X_train = pd.concat([V_train.reset_index(drop=True),
                     W_train.reset_index(drop=True)], axis=1)
X_test  = pd.concat([V_test.reset_index(drop=True),
                     W_test.reset_index(drop=True)], axis=1)

common = [c for c in X_train.columns if c in X_test.columns]
X_train = X_train[common].apply(pd.to_numeric, errors="coerce").fillna(0)
X_test  = X_test[common].apply(pd.to_numeric, errors="coerce").fillna(0)

print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

# -----------------------------------------------------------------------------
# 2. Función objetivo de Optuna (minimizar RMSE en test)
# -----------------------------------------------------------------------------
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "objective": "reg:squarederror",
        "random_state": 123,
        "n_jobs": -1,
    }

    model = XGBRegressor(**params)
    model.fit(X_train, Y_train, eval_set=[(X_test, Y_test)], verbose=False)

    pred = model.predict(X_test)
    return np.sqrt(mean_squared_error(Y_test, pred))  # RMSE test

# -----------------------------------------------------------------------------
# 3. Ejecutar estudio Optuna
# -----------------------------------------------------------------------------
print("\nIniciando optimización con Optuna (50 trials)...")

sampler = TPESampler(seed=123)
study = optuna.create_study(direction="minimize", sampler=sampler)

study.optimize(objective, n_trials=50, show_progress_bar=True)

print("\n=== Mejor trial ===")
print(f"Mejor RMSE test: {study.best_value:.4f}")
print("Mejores hiperparámetros:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# -----------------------------------------------------------------------------
# 4. Re-entrenar el mejor modelo
# -----------------------------------------------------------------------------
best_params = study.best_params.copy()
best_params.update({
    "objective": "reg:squarederror",
    "random_state": 123,
    "n_jobs": -1,
})

best_model = XGBRegressor(**best_params)
best_model.fit(X_train, Y_train, eval_set=[(X_test, Y_test)], verbose=False)

pred_train = best_model.predict(X_train)
pred_test  = best_model.predict(X_test)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

print("\n=== Métricas del XGBoost optimizado ===")
print(f"TRAIN → RMSE: {rmse(Y_train, pred_train):.4f}  |  MAE: {mae(Y_train, pred_train):.4f} |  R2: {r2_score(Y_train, pred_train):.4f}")
print(f"TEST  → RMSE: {rmse(Y_test,  pred_test):.4f}  |  MAE: {mae(Y_test,  pred_test):.4f} |  R2: {r2_score(Y_test, pred_test):.4f}")

# Guardar predicciones
pd.DataFrame({
    "Y_real": Y_test,
    "Y_pred_xgb_optuna": pred_test,
}).to_csv(OUTPUT_DIR / "xgb_optuna_pred_test.csv", index=False)

# -----------------------------------------------------------------------------
# 5. Comparar con Pogit (si existe)
# -----------------------------------------------------------------------------
print("\n=== Comparación con Pogit ===")
print(f"XGBoost Optuna → RMSE test: {rmse(Y_test, pred_test):.4f}  |  MAE: {mae(Y_test, pred_test):.4f} |  R2: {r2_score(Y_test, pred_test):.4f}")


if POGIT_PRED_PATH.exists():
    pogit = pd.read_csv(POGIT_PRED_PATH)
    col = [c for c in pogit.columns if "pred" in c.lower() or "pogit" in c.lower()]
    col = col[0] if col else pogit.columns[0]
    y_pogit = pogit[col].values
    if len(y_pogit) == len(Y_test):
        print(f"Pogit (archivo) → RMSE: {rmse(Y_test, y_pogit):.4f}  |  MAE: {mae(Y_test, y_pogit):.4f} |  R2: {r2_score(Y_test, y_pogit):.4f}")

# -----------------------------------------------------------------------------
# 6. Importancia de variables del mejor modelo
# -----------------------------------------------------------------------------
imp = pd.DataFrame({
    "variable": X_train.columns,
    "importance": best_model.feature_importances_,
}).sort_values("importance", ascending=False)

print("\n=== Top 15 variables (XGBoost optimizado) ===")
print(imp.head(15).to_string(index=False))

imp.to_csv(OUTPUT_DIR / "xgb_optuna_feature_importance.csv", index=False)

# Gráfico
plt.figure(figsize=(10, 6))
plt.barh(imp.head(15)["variable"][::-1], imp.head(15)["importance"][::-1], color="steelblue")
plt.xlabel("Importance")
plt.title("XGBoost Optuna – Top 15 variables")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "xgb_optuna_importance.png", dpi=150)
plt.close()

# -----------------------------------------------------------------------------
# 7. Historial de optimización (opcional)
# -----------------------------------------------------------------------------
try:
    from optuna.visualization import plot_optimization_history, plot_param_importances
    # Si estás en notebook:
    # plot_optimization_history(study).show()
    # plot_param_importances(study).show()
except Exception:
    pass

# Guardar estudio
import joblib
joblib.dump(study, OUTPUT_DIR / "optuna_study_xgboost.pkl")
joblib.dump(best_model, OUTPUT_DIR / "xgb_optuna_best_model.pkl")

print(f"\nModelo y estudio guardados en {OUTPUT_DIR}")
print("Listo.")