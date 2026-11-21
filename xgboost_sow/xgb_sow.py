"""
xgb_sow_two_part.py — Predicción óptima de SoW con XGBoost two-part
- Parte 1: P(SoW > 0)          → clasificador, optimiza AUC
- Parte 2: E[logit(SoW)|>0]    → regresor, optimiza MAE en logit
- Combinado: E[SoW] = P × expit(mu_logit)
- Optuna optimiza MAE final en SoW (no RMSE, no AUC aislado)
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, mean_absolute_error
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1) CONFIGURACIÓN
# ============================================================
TAG = "_min80"
PATH_XLSX = f"../data/matrices/matrices_data{TAG}.xlsx"
PATH_IDS_TRAIN = f"../data/matrices/ids_train{TAG}.csv"
PATH_IDS_TEST = f"../data/matrices/ids_test{TAG}.csv"
PATH_SOW_TRAIN = f"../data/matrices/sow_siow_train{TAG}.csv"
PATH_SOW_TEST = f"../data/matrices/sow_siow_test{TAG}.csv"
PATH_OUT = f"../data/matrices/xgb_sow_two_part_test{TAG}.csv"
PATH_PARAMS = f"../data/matrices/xgb_sow_two_part_best_params{TAG}.json"

RANDOM_STATE = 42
N_TRIALS = 80
N_FOLDS = 5
EPS = 1e-6  # para clipping del logit

# ============================================================
# 2) CARGA Y ALINEACIÓN
# ============================================================
V_train = pd.read_excel(PATH_XLSX, sheet_name="V_train").drop(columns=lambda c: c.startswith("Intercept"), errors="ignore")
V_test  = pd.read_excel(PATH_XLSX, sheet_name="V_test").drop(columns=lambda c: c.startswith("Intercept"), errors="ignore")
W_train = pd.read_excel(PATH_XLSX, sheet_name="W_train").drop(columns=lambda c: c.startswith("Intercept"), errors="ignore")
W_test  = pd.read_excel(PATH_XLSX, sheet_name="W_test").drop(columns=lambda c: c.startswith("Intercept"), errors="ignore")

X_train = pd.concat([V_train, W_train], axis=1)
X_test  = pd.concat([V_test,  W_test ], axis=1)

# Alinear columnas
common = X_train.columns.intersection(X_test.columns)
X_train = X_train[common].apply(pd.to_numeric, errors="coerce").fillna(0)
X_test  = X_test[common].apply(pd.to_numeric, errors="coerce").fillna(0)

ids_train = pd.read_csv(PATH_IDS_TRAIN)["response_id"].astype(str).values
ids_test  = pd.read_csv(PATH_IDS_TEST)["response_id"].astype(str).values

sow_train_df = pd.read_csv(PATH_SOW_TRAIN)
sow_test_df  = pd.read_csv(PATH_SOW_TEST)

# Alinear por response_id
sow_train_df["response_id"] = sow_train_df["response_id"].astype(str)
sow_test_df ["response_id"] = sow_test_df ["response_id"].astype(str)

sow_train_df = sow_train_df.set_index("response_id").loc[ids_train].reset_index()
sow_test_df  = sow_test_df .set_index("response_id").loc[ids_test ].reset_index()

y_train = sow_train_df["sow"].values
y_test  = sow_test_df ["sow"].values

assert len(y_train) == len(X_train), f"Mismatch: {len(y_train)} vs {len(X_train)}"
assert len(y_test)  == len(X_test),  f"Mismatch: {len(y_test)} vs {len(X_test)}"

print(f"TRAIN: n={len(y_train)} | %ceros={100*(y_train==0).mean():.1f}%")
print(f"TEST : n={len(y_test)}  | %ceros={100*(y_test ==0).mean():.1f}%")
print(f"Features: {X_train.shape[1]}")

# ============================================================
# 3) HELPERS
# ============================================================
def safe_logit(y):
    """logit con clipping para evitar infinitos."""
    y_clipped = np.clip(y, EPS, 1 - EPS)
    return np.log(y_clipped / (1 - y_clipped))

def two_part_predict(clf, reg, X):
    """E[SoW] = P(SoW>0) × expit(E[logit(SoW)|SoW>0])."""
    p_pos  = clf.predict_proba(X)[:, 1]
    mu_pos = expit(reg.predict(X))
    pred   = np.clip(p_pos * mu_pos, 0, 1)
    return pred, p_pos, mu_pos

def metrics_sow(y_true, y_pred, p_pos=None):
    """Métricas correctas para SoW two-part."""
    out = {
        "RMSE":     float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "MAE":      float(mean_absolute_error(y_true, y_pred)),
        "Spearman": float(spearmanr(y_true, y_pred).correlation),
        "R2":       float(1 - np.sum((y_true - y_pred)**2) /
                          np.sum((y_true - y_true.mean())**2)),
        "Mean_real": float(y_true.mean()),
        "Mean_pred": float(y_pred.mean()),
    }
    if p_pos is not None:
        mask = y_true > 0
        if mask.sum() > 10 and len(np.unique((y_true>0).astype(int))) > 1:
            out["AUC"] = float(roc_auc_score((y_true > 0).astype(int), p_pos))
    return out

# ============================================================
# 4) OPTUNA: objetivo = MAE final en SoW (5-fold CV)
# ============================================================
def objective(trial):
    # ---- Híper para parte 1 (clasificador)
    params_clf = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "max_depth":        trial.suggest_int("clf_max_depth", 2, 6),
        "learning_rate":    trial.suggest_float("clf_lr", 0.01, 0.15, log=True),
        "n_estimators":     trial.suggest_int("clf_n_estimators", 100, 800, step=50),
        "min_child_weight": trial.suggest_float("clf_mcw", 1, 20, log=True),
        "subsample":        trial.suggest_float("clf_subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("clf_colsample", 0.5, 1.0),
        "reg_lambda":       trial.suggest_float("clf_lambda", 1e-2, 10, log=True),
        "reg_alpha":        trial.suggest_float("clf_alpha", 1e-4, 1, log=True),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": 0,
    }
    # ---- Híper para parte 2 (regresor sobre logit)
    params_reg = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "tree_method": "hist",
        "max_depth":        trial.suggest_int("reg_max_depth", 2, 6),
        "learning_rate":    trial.suggest_float("reg_lr", 0.01, 0.15, log=True),
        "n_estimators":     trial.suggest_int("reg_n_estimators", 100, 800, step=50),
        "min_child_weight": trial.suggest_float("reg_mcw", 1, 20, log=True),
        "subsample":        trial.suggest_float("reg_subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("reg_colsample", 0.5, 1.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-2, 10, log=True),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 1, log=True),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": 0,
    }

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    y_bin = (y_train > 0).astype(int)
    oof_pred = np.zeros(len(y_train))

    for tr_idx, va_idx in cv.split(X_train, y_bin):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        # ---- Parte 1
        clf = xgb.XGBClassifier(**params_clf,
                                early_stopping_rounds=40)
        clf.fit(X_tr, (y_tr > 0).astype(int),
                eval_set=[(X_va, (y_va > 0).astype(int))],
                verbose=False)

        # ---- Parte 2 (solo positivos)
        pos_tr = y_tr > 0
        pos_va = y_va > 0
        if pos_tr.sum() < 20 or pos_va.sum() < 5:
            oof_pred[va_idx] = 0
            continue

        reg = xgb.XGBRegressor(**params_reg,
                               early_stopping_rounds=40)
        reg.fit(X_tr[pos_tr], safe_logit(y_tr[pos_tr]),
                eval_set=[(X_va[pos_va], safe_logit(y_va[pos_va]))],
                verbose=False)

        # ---- Combinar
        p_pos  = clf.predict_proba(X_va)[:, 1]
        mu_pos = expit(reg.predict(X_va))
        oof_pred[va_idx] = np.clip(p_pos * mu_pos, 0, 1)

    # Métrica de optimización: MAE en SoW
    return float(mean_absolute_error(y_train, oof_pred))

# ============================================================
# 5) CORRER OPTUNA
# ============================================================
print(f"\nOptuna: {N_TRIALS} trials, {N_FOLDS}-fold CV, objetivo = MAE(SoW)")
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best = study.best_params
print(f"\nMejor MAE CV: {study.best_value:.4f}")
print(f"Mejores params: {best}")

import json
with open(PATH_PARAMS, "w") as f:
    json.dump({"best_params": best, "best_mae": study.best_value}, f, indent=2)

# ============================================================
# 6) REENTRENAR EN TRAIN COMPLETO Y EVALUAR EN TEST
# ============================================================
params_clf = {
    "objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist",
    "max_depth":        best["clf_max_depth"],
    "learning_rate":    best["clf_lr"],
    "n_estimators":     best["clf_n_estimators"],
    "min_child_weight": best["clf_mcw"],
    "subsample":        best["clf_subsample"],
    "colsample_bytree": best["clf_colsample"],
    "reg_lambda":       best["clf_lambda"],
    "reg_alpha":        best["clf_alpha"],
    "random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": 0,
}
params_reg = {
    "objective": "reg:squarederror", "eval_metric": "mae", "tree_method": "hist",
    "max_depth":        best["reg_max_depth"],
    "learning_rate":    best["reg_lr"],
    "n_estimators":     best["reg_n_estimators"],
    "min_child_weight": best["reg_mcw"],
    "subsample":        best["reg_subsample"],
    "colsample_bytree": best["reg_colsample"],
    "reg_lambda":       best["reg_lambda"],
    "reg_alpha":        best["reg_alpha"],
    "random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": 0,
}

# Split interno train/val (10%) para early stopping del fit final
from sklearn.model_selection import train_test_split
idx_tr, idx_va = train_test_split(
    np.arange(len(y_train)), test_size=0.10,
    stratify=(y_train > 0).astype(int), random_state=RANDOM_STATE,
)

print("\nEntrenando modelo final (train completo)...")
clf_final = xgb.XGBClassifier(**params_clf, early_stopping_rounds=40)
clf_final.fit(X_train.iloc[idx_tr], (y_train[idx_tr] > 0).astype(int),
              eval_set=[(X_train.iloc[idx_va], (y_train[idx_va] > 0).astype(int))],
              verbose=False)

pos_tr = y_train[idx_tr] > 0
pos_va = y_train[idx_va] > 0
reg_final = xgb.XGBRegressor(**params_reg, early_stopping_rounds=40)
reg_final.fit(X_train.iloc[idx_tr][pos_tr], safe_logit(y_train[idx_tr][pos_tr]),
              eval_set=[(X_train.iloc[idx_va][pos_va], safe_logit(y_train[idx_va][pos_va]))],
              verbose=False)

# ---- Predicciones
pred_train, p_pos_train, mu_pos_train = two_part_predict(clf_final, reg_final, X_train)
pred_test,  p_pos_test,  mu_pos_test  = two_part_predict(clf_final, reg_final, X_test)

# ---- Métricas
print("\n" + "="*60)
print("MÉTRICAS — XGBoost Two-Part SoW")
print("="*60)
m_train = metrics_sow(y_train, pred_train, p_pos_train)
m_test  = metrics_sow(y_test,  pred_test,  p_pos_test)

for name, m in [("TRAIN", m_train), ("TEST", m_test)]:
    print(f"\n{name}:")
    for k, v in m.items():
        print(f"  {k:12s} = {v:.4f}")

# ---- Baseline (media de train)
baseline_test = np.full_like(y_test, y_train.mean())
m_base = metrics_sow(y_test, baseline_test)
print(f"\nBASELINE (media train = {y_train.mean():.4f}):")
for k, v in m_base.items():
    print(f"  {k:12s} = {v:.4f}")

# ---- Comparación de mejora
print("\n" + "="*60)
print("MEJORA vs BASELINE")
print("="*60)
print(f"  MAE:      {m_base['MAE']:.4f} → {m_test['MAE']:.4f}  "
      f"({100*(m_base['MAE']-m_test['MAE'])/m_base['MAE']:+.1f}%)")
print(f"  RMSE:     {m_base['RMSE']:.4f} → {m_test['RMSE']:.4f}  "
      f"({100*(m_base['RMSE']-m_test['RMSE'])/m_base['RMSE']:+.1f}%)")
print(f"  Spearman: {m_base['Spearman']:.4f} → {m_test['Spearman']:.4f}")
print(f"  R²:       {m_base['R2']:.4f} → {m_test['R2']:.4f}")

# ============================================================
# 7) GUARDAR PREDICCIONES
# ============================================================
out = pd.DataFrame({
    "response_id":   ids_test,
    "SoW_real":      y_test,
    "SoW_pred":      pred_test,
    "P_pos":         p_pos_test,
    "E_pos_logit":   mu_pos_test,
    "SoW_pred_train": pred_train[:len(ids_test)] if len(pred_train) == len(ids_test) else np.nan,
})
out.to_csv(PATH_OUT, index=False)
print(f"\nGuardado: {PATH_OUT}")

# ============================================================
# 8) DIAGNÓSTICO RÁPIDO
# ============================================================
print("\n" + "="*60)
print("DIAGNÓSTICO")
print("="*60)

# Distribución de predicciones vs real
print(f"\nSoW_real  : min={y_test.min():.4f}  max={y_test.max():.4f}  "
      f"mean={y_test.mean():.4f}  sd={y_test.std():.4f}")
print(f"SoW_pred  : min={pred_test.min():.4f}  max={pred_test.max():.4f}  "
      f"mean={pred_test.mean():.4f}  sd={pred_test.std():.4f}")
print(f"→ El ratio sd_pred/sd_real debe ser > 0.5 (antes era 0.05)")

# Calibración por deciles de predicción
print("\nCalibración por deciles de SoW_pred:")
cal = pd.DataFrame({"y": y_test, "pred": pred_test})
cal["dec"] = pd.qcut(cal["pred"], 10, labels=False, duplicates="drop")
cal_summary = cal.groupby("dec").agg(
    n=("y", "size"),
    pred_mean=("pred", "mean"),
    real_mean=("y", "mean"),
    pct_pos=("y", lambda x: (x > 0).mean()),
).round(3)
print(cal_summary.to_string())

# Top-20 clientes por predicción (para uso comercial)
print("\nTop-20 clientes por SoW_pred:")
top20 = out.nlargest(20, "SoW_pred")[["response_id", "SoW_real", "SoW_pred", "P_pos"]]
print(top20.to_string(index=False))