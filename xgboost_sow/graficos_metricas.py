import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ------------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------------
ARCHIVO = "../data/matrices/metricas tesis.xlsx"   # ajusta la ruta si es necesario

# Hoja -> título del gráfico
HOJAS = {
    "share of wallet (sow)":    "Share of Wallet (SoW)",
    "size of wallet(siow)":     "Size of Wallet (SioW)",
    "gasto discretizado(Y)":    "Gasto discretizado (Y)",
}

COL_X, COL_MODELO = "Umbral MIN_MONETARY_T1", "Modelo"
COL_RMSE, COL_MAE = "TEST RMSE", "TEST MAE"

# En la hoja SoW el valor de xgboost con umbral 20 aparece como 18.24
# (probablemente es 0.1824 por un error de digitación y distorsiona el eje).
# Pon True para corregirlo automáticamente al graficar (no modifica el Excel).
CORREGIR_OUTLIER_SOW = False

COLORES = {"RMSE": "tab:blue", "MAE": "tab:red"}   # color por métrica (cada eje)
ESTILOS = ["-", "--", ":", "-."]                   # estilo de línea por modelo
MARCADORES = ["o", "s", "^", "D"]                  # marcador por modelo


# ------------------------------------------------------------------
# CARGA Y NORMALIZACIÓN
# ------------------------------------------------------------------
def cargar_hoja(ruta, hoja):
    df = pd.read_excel(ruta, sheet_name=hoja)
    df.columns = [str(c).strip() for c in df.columns]

    # La hoja Y no tiene la columna de umbral con nombre ni columna de modelo
    if COL_X not in df.columns:
        df = df.rename(columns={df.columns[0]: COL_X})
    if COL_MODELO not in df.columns:
        # buscar variantes (p. ej. "MODELO")
        alt = [c for c in df.columns if c.lower() == "modelo"]
        if alt:
            df = df.rename(columns={alt[0]: COL_MODELO})
        else:
            df[COL_MODELO] = "Modelo"          # una sola serie

    df[COL_MODELO] = df[COL_MODELO].astype(str).str.strip()
    return df[[COL_X, COL_MODELO, COL_RMSE, COL_MAE]].dropna(subset=[COL_X])


# ------------------------------------------------------------------
# GRÁFICO
# ------------------------------------------------------------------
def graficar(df, titulo, archivo_salida=None):
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax2 = ax1.twinx()                          # segundo eje Y

    modelos = list(df[COL_MODELO].unique())
    for i, modelo in enumerate(modelos):
        d = df[df[COL_MODELO] == modelo].sort_values(COL_X)
        ls, mk = ESTILOS[i % len(ESTILOS)], MARCADORES[i % len(MARCADORES)]

        ax1.plot(d[COL_X], d[COL_RMSE], color=COLORES["RMSE"],
                 linestyle=ls, marker=mk, label=f"{modelo} – RMSE")
        ax2.plot(d[COL_X], d[COL_MAE], color=COLORES["MAE"],
                 linestyle=ls, marker=mk, label=f"{modelo} – MAE")

    ax1.set_xlabel("Umbral MIN_MONETARY_T1")
    ax1.set_ylabel("TEST RMSE", color=COLORES["RMSE"])
    ax2.set_ylabel("TEST MAE", color=COLORES["MAE"])
    ax1.tick_params(axis="y", colors=COLORES["RMSE"])
    ax2.tick_params(axis="y", colors=COLORES["MAE"])
    ax1.set_xticks(sorted(df[COL_X].unique()))
    ax1.grid(alpha=0.3)
    ax1.set_title(titulo)

    # Leyenda combinada de ambos ejes
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper center",
               bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

    fig.tight_layout()
    if archivo_salida:
        fig.savefig(archivo_salida, dpi=200, bbox_inches="tight")
    return fig


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
if __name__ == "__main__":
    for hoja, titulo in HOJAS.items():
        df = cargar_hoja(ARCHIVO, hoja)

        if CORREGIR_OUTLIER_SOW and hoja.startswith("share of wallet"):
            mask = df[COL_RMSE] > 5
            if mask.any():
                print(f"[aviso] {mask.sum()} valor(es) de RMSE > 5 en '{hoja}' "
                      f"divididos entre 100:\n{df[mask]}")
                df.loc[mask, COL_RMSE] = df.loc[mask, COL_RMSE] / 100

        nombre = hoja.split("(")[-1].replace(")", "").strip()
        graficar(df, f"{titulo}: RMSE y MAE vs. umbral", f"grafico_{nombre}.png")

    plt.show()
