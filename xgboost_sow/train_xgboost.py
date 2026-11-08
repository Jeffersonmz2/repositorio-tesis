# =============================================================================
# XGBoost: V + W → predecir Y
# Flujo: entrenar → RMSE/MAE → comparar con Pogit → importancia de variables
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# Rutas
# -----------------------------------------------------------------------------
TAG="_min30"
DATA_DIR = Path("../data/matrices")
WB = DATA_DIR / f"matrices_data{TAG}.xlsx"
POGIT_PRED_PATH = DATA_DIR /"pogit_pred_test.csv"   # opcional (exportado desde R)
OUTPUT_DIR = DATA_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Cargar matrices
# -----------------------------------------------------------------------------
print("Cargando matrices...")

V_train = pd.read_excel(WB, sheet_name="V_train")
V_test  = pd.read_excel(WB, sheet_name="V_test")
W_train = pd.read_excel(WB, sheet_name="W_train")
W_test  = pd.read_excel(WB, sheet_name="W_test")
Y_train = pd.read_excel(WB, sheet_name="Y_train").iloc[:, 0].astype(float)
Y_test  = pd.read_excel(WB, sheet_name="Y_test").iloc[:, 0].astype(float)

# Quitar Intercept (XGBoost no lo necesita)
for df in (V_train, V_test, W_train, W_test):
    cols = [c for c in df.columns if str(c).lower().startswith("intercept")]
    if cols:
        df.drop(columns=cols, inplace=True)

# Unir V + W
X_train = pd.concat([V_train.reset_index(drop=True),
                     W_train.reset_index(drop=True)], axis=1)
X_test  = pd.concat([V_test.reset_index(drop=True),
                     W_test.reset_index(drop=True)], axis=1)

# Misma columnas y orden
common = [c for c in X_train.columns if c in X_test.columns]
X_train = X_train[common].apply(pd.to_numeric, errors="coerce").fillna(0)
X_test  = X_test[common].apply(pd.to_numeric, errors="coerce").fillna(0)

print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
print(f"Y_train: {Y_train.shape} | Y_test: {Y_test.shape}")

# -----------------------------------------------------------------------------
# 2. Entrenar XGBoost (regresión sobre Y)
# -----------------------------------------------------------------------------
print("\nEntrenando XGBoost...")

model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    min_child_weight=5,
    objective="reg:squarederror",
    random_state=123,
    n_jobs=-1,
)

model.fit(
    X_train, Y_train,
    eval_set=[(X_test, Y_test)],
    verbose=False,
)

# -----------------------------------------------------------------------------
# 3. Predicciones y métricas
# -----------------------------------------------------------------------------
pred_train = model.predict(X_train)
pred_test  = model.predict(X_test)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

print("\n=== Métricas XGBoost ===")
print(f"TRAIN → RMSE: {rmse(Y_train, pred_train):.4f}  |  MAE: {mae(Y_train, pred_train):.4f}")
print(f"TEST  → RMSE: {rmse(Y_test,  pred_test):.4f}  |  MAE: {mae(Y_test,  pred_test):.4f}")

# Guardar predicciones de test
pred_df = pd.DataFrame({
    "Y_real": Y_test.values,
    "Y_pred_xgb": pred_test,
})
pred_df.to_csv(OUTPUT_DIR / "xgb_pred_test.csv", index=False)
print(f"\nPredicciones guardadas en {OUTPUT_DIR / 'xgb_pred_test.csv'}")

# -----------------------------------------------------------------------------
# 4. Comparar con Pogit (si existe archivo exportado desde R)
# -----------------------------------------------------------------------------
# En R, después del Pogit, exporta así:
# write.csv(data.frame(Y_pred_pogit = pred_test), "data/matrices/pogit_pred_test.csv", row.names = FALSE)

if POGIT_PRED_PATH.exists():
    pogit = pd.read_csv(POGIT_PRED_PATH)
    # Ajustar nombre de columna si es necesario
    col = [c for c in pogit.columns if "pred" in c.lower() or "pogit" in c.lower()]
    if not col:
        col = [pogit.columns[0]]
    y_pogit = pogit[col[0]].values

    if len(y_pogit) == len(Y_test):
        print("\n=== Comparación XGBoost vs Pogit (TEST) ===")
        print(f"XGBoost → RMSE: {rmse(Y_test, pred_test):.4f}  |  MAE: {mae(Y_test, pred_test):.4f}")
        print(f"Pogit   → RMSE: {rmse(Y_test, y_pogit):.4f}  |  MAE: {mae(Y_test, y_pogit):.4f}")
    else:
        print("\n[Aviso] El archivo de predicciones Pogit no tiene el mismo tamaño que Y_test.")
else:
    print("\n[Info] No se encontró pogit_pred_test.csv.")
    print("       Exporta desde R las predicciones de test del Pogit para comparar.")

# -----------------------------------------------------------------------------
# 5. Importancia de variables (para refinar el Pogit)
# -----------------------------------------------------------------------------
print("\n=== Top 20 variables más importantes (XGBoost) ===")

imp = pd.DataFrame({
    "variable": X_train.columns,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

print(imp.head(20).to_string(index=False))

# Guardar importancia completa
imp.to_csv(OUTPUT_DIR / "xgb_feature_importance.csv", index=False)
print(f"\nImportancia guardada en {OUTPUT_DIR / 'xgb_feature_importance.csv'}")

# -----------------------------------------------------------------------------
# 6. Gráfico de importancia
# -----------------------------------------------------------------------------
top_n = 20
imp_top = imp.head(top_n)

plt.figure(figsize=(10, 7))
sns.barplot(data=imp_top, x="importance", y="variable", color="steelblue")
plt.title("XGBoost – Importancia de variables (Top 20)")
plt.xlabel("Importance (gain)")
plt.ylabel("")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "xgb_feature_importance.png", dpi=150)
plt.close()
print(f"Gráfico guardado en {OUTPUT_DIR / 'xgb_feature_importance.png'}")

# -----------------------------------------------------------------------------
# 7. Sugerencia automática de variables para el Pogit
# -----------------------------------------------------------------------------
# Quédate con las que acumulan ~90% de la importancia
imp["cumsum"] = imp["importance"].cumsum() / imp["importance"].sum()
keep = imp.loc[imp["cumsum"] <= 0.90, "variable"].tolist()
# Asegurar que al menos haya unas cuantas
if len(keep) < 5:
    keep = imp.head(10)["variable"].tolist()

print("\n=== Variables sugeridas para refinar el Pogit (≈90% importancia) ===")
for v in keep:
    print(f"  - {v}")

# Separar sugerencia para parte de ceros (comportamiento) vs conteo
behavior_keywords = ["recency", "frequency", "monetary", "dollar", "product", "month", "n_states"]
zero_candidates = [v for v in keep if any(k in v.lower() for k in behavior_keywords)]
count_candidates = keep  # todas las importantes para la parte de intensidad

print("\nSugerencia parte LOGIT (ceros) del Pogit:")
print("  ", zero_candidates if zero_candidates else keep[:5])

print("\nSugerencia parte POISSON (conteo) del Pogit:")
print("  ", count_candidates)

print("\nListo.")