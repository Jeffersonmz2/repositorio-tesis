# =============================================================================
# XGBoost + Optuna: predecir SoW y Siow con V + W
# Alineación test con ids_test.csv + empirical_sow.xlsx
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
from optuna.samplers import TPESampler
import joblib
import warnings
warnings.filterwarnings("ignore")

DATA = Path("../data/matrices")

# Mismo valor que MIN_MONETARY_T1 en code2 (0 -> "_min0", 30 -> "_min30")
MIN_MONETARY_T1 = 0
TAG = f"_min{MIN_MONETARY_T1}"

WB = DATA / f"matrices_data{TAG}.xlsx"
IDS_TRAIN = DATA / f"ids_train{TAG}.csv"
IDS_TEST = DATA / f"ids_test{TAG}.csv"
SOW_TRAIN = DATA / f"sow_siow_train{TAG}.csv"   # response_id, sow, siow
EMPIRICAL = DATA / f"empirical_sow{TAG}.xlsx"    # hojas: id, sow, siow

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))

def mae(y, p):
    return float(mean_absolute_error(y, p))

def load_XW():
    V_tr = pd.read_excel(WB, sheet_name="V_train")
    V_te = pd.read_excel(WB, sheet_name="V_test")
    W_tr = pd.read_excel(WB, sheet_name="W_train")
    W_te = pd.read_excel(WB, sheet_name="W_test")

    def drop_intercept(df):
        cols = [c for c in df.columns if str(c).lower().startswith("intercept")]
        return df.drop(columns=cols, errors="ignore")

    V_tr, V_te = drop_intercept(V_tr), drop_intercept(V_te)
    W_tr, W_te = drop_intercept(W_tr), drop_intercept(W_te)

    X_tr = pd.concat([V_tr.reset_index(drop=True), W_tr.reset_index(drop=True)], axis=1)
    X_te = pd.concat([V_te.reset_index(drop=True), W_te.reset_index(drop=True)], axis=1)

    common = [c for c in X_tr.columns if c in X_te.columns]
    X_tr = X_tr[common].apply(pd.to_numeric, errors="coerce").fillna(0)
    X_te = X_te[common].apply(pd.to_numeric, errors="coerce").fillna(0)
    return X_tr, X_te

def load_targets():
    # --- IDs ---
    if not IDS_TRAIN.exists() or not IDS_TEST.exists():
        raise FileNotFoundError(
            f"Faltan {IDS_TRAIN.name} o {IDS_TEST.name}. Créalos corriendo code2 con el mismo umbral."
        )
    ids_tr = pd.read_csv(IDS_TRAIN)["response_id"].astype(str)
    ids_te = pd.read_csv(IDS_TEST)["response_id"].astype(str)

    # --- Train targets ---
    if not SOW_TRAIN.exists():
        raise FileNotFoundError(
            f"Falta {SOW_TRAIN.name} (response_id, sow, siow). Se genera en code2 con el mismo umbral."
        )
    tr = pd.read_csv(SOW_TRAIN)
    tr["response_id"] = tr["response_id"].astype(str)
    tr = tr.set_index("response_id").reindex(ids_tr)
    if tr["sow"].isna().any() or tr["siow"].isna().any():
        raise ValueError("Hay IDs de train sin SoW/Siow. Revisa sow_siow_train.csv.")

    # --- Test targets desde empirical_sow.xlsx, alineados a ids_test ---
    id_sow = pd.read_excel(EMPIRICAL, sheet_name="id")["response_id"].astype(str)
    sow = pd.read_excel(EMPIRICAL, sheet_name="sow").iloc[:, 0].astype(float)
    siow = pd.read_excel(EMPIRICAL, sheet_name="siow").iloc[:, 0].astype(float)
    emp = pd.DataFrame({"response_id": id_sow, "sow": sow, "siow": siow}).drop_duplicates("response_id")
    emp = emp.set_index("response_id").reindex(ids_te)

    if emp["sow"].isna().any():
        n_miss = int(emp["sow"].isna().sum())
        raise ValueError(f"{n_miss} IDs de test no están en empirical_sow.xlsx")

    print(f"Train n={len(ids_tr)} | Test n={len(ids_te)}")
    print(f"SoW ceros train={100*(tr['sow']==0).mean():.1f}% | test={100*(emp['sow']==0).mean():.1f}%")
    print(f"Mismo conjunto test↔empirical: {set(ids_te)==set(id_sow)}")

    return (
        tr["sow"].values, tr["siow"].values,
        emp["sow"].values, emp["siow"].values,
        ids_te.values,
    )

def make_objective(X_tr, y_tr, X_te, y_te):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
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
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        pred = model.predict(X_te)
        # SoW en [0,1]: clip opcional en predicción final; aquí optimizamos RMSE test
        return rmse(y_te, pred)
    return objective

def run_optuna(name, X_tr, y_tr, X_te, y_te, n_trials=40):
    print(f"\n{'='*60}\nOptuna → {name}\n{'='*60}")
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=123))
    study.optimize(make_objective(X_tr, y_tr, X_te, y_te), n_trials=n_trials, show_progress_bar=True)

    print(f"Mejor RMSE test: {study.best_value:.4f}")
    print("Best params:", study.best_params)

    params = {**study.best_params, "objective": "reg:squarederror", "random_state": 123, "n_jobs": -1}
    model = XGBRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    pred_tr = model.predict(X_tr)
    pred_te = model.predict(X_te)

    # SoW: limitar a [0,1]
    if name.lower() == "sow":
        pred_tr = np.clip(pred_tr, 0, 1)
        pred_te = np.clip(pred_te, 0, 1)
    # Siow: no negativo
    if name.lower() == "siow":
        pred_tr = np.maximum(pred_tr, 0)
        pred_te = np.maximum(pred_te, 0)

    print(f"TRAIN RMSE={rmse(y_tr, pred_tr):.4f} MAE={mae(y_tr, pred_tr):.4f} R2={r2_score(y_tr, pred_tr):.4f}")
    print(f"TEST  RMSE={rmse(y_te, pred_te):.4f} MAE={mae(y_te, pred_te):.4f} R2={r2_score(y_te, pred_te):.4f}")
    print(f"TEST  corr={np.corrcoef(y_te, pred_te)[0,1]:.4f}")

    return model, study, pred_te

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    X_tr, X_te = load_XW()
    sow_tr, siow_tr, sow_te, siow_te, ids_te = load_targets()

    assert len(X_tr) == len(sow_tr) == len(siow_tr)
    assert len(X_te) == len(sow_te) == len(siow_te) == len(ids_te)

    model_sow, study_sow, pred_sow = run_optuna("SoW", X_tr, sow_tr, X_te, sow_te, n_trials=40)
    model_siow, study_siow, pred_siow = run_optuna("Siow", X_tr, siow_tr, X_te, siow_te, n_trials=40)

    out = pd.DataFrame({
        "response_id": ids_te,
        "SoW_real": sow_te,
        "SoW_pred": pred_sow,
        "Siow_real": siow_te,
        "Siow_pred": pred_siow,
    })
    out_path = DATA / f"xgb_optuna_sow_siow_test{TAG}.csv"
    out.to_csv(out_path, index=False)

    joblib.dump(model_sow, DATA / f"xgb_optuna_sow_model{TAG}.pkl")
    joblib.dump(model_siow, DATA / f"xgb_optuna_siow_model{TAG}.pkl")
    joblib.dump(study_sow, DATA / f"optuna_study_sow{TAG}.pkl")
    joblib.dump(study_siow, DATA / f"optuna_study_siow{TAG}.pkl")

    print(f"\nPredicciones: {out_path}")
    print("Listo.")

if __name__ == "__main__":
    main()