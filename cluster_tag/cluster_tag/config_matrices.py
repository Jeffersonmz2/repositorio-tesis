"""
Configuración compartida de los notebooks K-Means (code1 ... code6).

* MIN_MONETARY_T1 -> TAG: discrimina las matrices de entrada y TODAS las
  carpetas/archivos de resultados. Para cambiar de umbral, edita solo aquí.
* Entrada: data/matrices/matrices_data{TAG}.csv  (o .xlsx con el mismo nombre),
  que contiene las matrices de TRAIN de W (transaccionales) y V (perfilamiento).
"""
import warnings
from pathlib import Path

import pandas as pd

# ------------------------------------------------------------------
# 1. Umbral monetario en T1 y TAG
# ------------------------------------------------------------------
MIN_MONETARY_T1 = 80
TAG = f"_min{MIN_MONETARY_T1}"           # 0 -> "_min0", 30 -> "_min30"

# Ruta por defecto (relativa a la carpeta del notebook); es la que usan los notebooks:
#   RUTA_CSV_MANUAL = f"../../data/matrices/matrices_data{TAG}.csv"
RUTA_MATRICES_DEFAULT = f"../../data/matrices/matrices_data{TAG}.csv"

# Los resultados NO se escriben en data/matrices: van a esta carpeta, en
# subcarpetas con el TAG (resultados_..._min0, etc.).
CARPETA_SALIDA = Path.cwd()

# ------------------------------------------------------------------
# 2. Nombres de columnas
# ------------------------------------------------------------------
W_VARS = [
    "recency_days_2018_Samsung",
    "frequency_2018_Samsung",
    "monetary_2018_Samsung",
    "dollar_avg_12m",
    "dollar_top1_12m",
    "dollar_avg12_top1_ratio",
]
HOJA_W_TRAIN = "W_train"
HOJA_V_TRAIN = "V_train"


# ------------------------------------------------------------------
# 3. Localización del archivo
# ------------------------------------------------------------------
def ruta_matrices(ruta=None) -> Path:
    """Devuelve la ruta del archivo de matrices; si no existe el .csv prueba el .xlsx."""
    ruta = Path(RUTA_MATRICES_DEFAULT if ruta is None else ruta).expanduser()
    intentos = [ruta]
    if ruta.suffix.lower() == ".csv":
        intentos.append(ruta.with_suffix(".xlsx"))
    for p in intentos:
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError(
        f"No se encontró el archivo de matrices para el TAG {TAG!r}. Probé:\n  "
        + "\n  ".join(str(p.resolve()) for p in intentos)
        + f"\n(carpeta de trabajo actual: {Path.cwd()})"
    )


# ------------------------------------------------------------------
# 4. Carga de matrices
# ------------------------------------------------------------------
def _leer(ruta, hoja):
    ruta = Path(ruta)
    if ruta.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(ruta, sheet_name=hoja)
    else:
        df = pd.read_csv(ruta, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _es_intercepto(c: str) -> bool:
    return c.lower().startswith("intercept")


def _asegurar_response_id(df: pd.DataFrame) -> pd.DataFrame:
    if "response_id" not in df.columns:
        warnings.warn(
            "La matriz no trae 'response_id': se crea un identificador por posición "
            "(train_00001, ...). Sirve para K-Means, pero NO permite cruzar con otras "
            "tablas; el orden de filas debe ser el de train_ids.",
            stacklevel=3,
        )
        df.insert(0, "response_id", [f"train_{i + 1:05d}" for i in range(len(df))])
    df["response_id"] = df["response_id"].astype("string")
    return df


def cargar_W_train(ruta) -> pd.DataFrame:
    """
    W de TRAIN listo para los notebooks: response_id + las 6 variables W.
    Funciona con el .xlsx (hoja W_train) y con un .csv que contenga W (y V) unidas.
    El Intercept se descarta (es constante y rompería auditorías/escalado).
    """
    df = _leer(ruta, HOJA_W_TRAIN)
    faltan = [c for c in W_VARS if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas de W en {Path(ruta).name}: {faltan}")
    df = _asegurar_response_id(df)
    return df[["response_id"] + W_VARS].copy()


def cargar_V_train(ruta) -> pd.DataFrame:
    """V de TRAIN (perfilamiento). Sin Intercept; con response_id si existe en el archivo."""
    df = _leer(ruta, HOJA_V_TRAIN)
    if Path(ruta).suffix.lower() == ".csv":   # unión V+W(+Y): quedarse solo con V
        excluir = set(W_VARS) | {"response_id"}
        df = df[[c for c in df.columns
                 if c not in excluir and not c.lower().startswith("y_")]]
    df = df[[c for c in df.columns if not _es_intercepto(c)]].copy()
    return df


if __name__ == "__main__":
    p = ruta_matrices()
    w = cargar_W_train(p)
    print("MIN_MONETARY_T1:", MIN_MONETARY_T1, "| TAG:", TAG)
    print("Archivo:", p)
    print("W_train:", w.shape)
