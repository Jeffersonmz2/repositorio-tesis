"""
Patrones de datos faltantes de la base Purchases (segmento Samsung).

Entrada : CSV con las columnas ORIGINALES de Purchases para el segmento elegido.
Salidas : figura PNG (idéntica a la Figura 6.1), tabla de patrones y resumen por variable.
"""
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# ------------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------------
RUTA_CSV = Path(r"..\data\survey-data\amazon-purchases.csv")
NOMBRE_SEGMENTO = "Purchases_Samsung"          # aparece en el título del eje X
SALIDA = Path("faltantes_purchases_samsung")
TRATAR_VACIO_COMO_FALTANTE = True              # "" o solo espacios cuentan como faltante

# Etiquetas normalizadas (en minúsculas y con guiones bajos)
ETIQUETAS = {
    "asin_isbn_product_code": "ASIN/ISBN (Product Code)",
    "category": "Category",
    "shipping_address_state": "Shipping Address State",
}
COLOR_PRESENTE = "#E2E8F0"
COLOR_FALTANTE = "#DC3A58"


# ------------------------------------------------------------------
# CÁLCULO
# ------------------------------------------------------------------
def normalizar_nombre(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", col.strip().lower()).strip("_")


def mascara_faltantes(df: pd.DataFrame) -> pd.DataFrame:
    m = df.isna()
    if TRATAR_VACIO_COMO_FALTANTE:
        for c in df.columns:
            if df[c].dtype == object or str(df[c].dtype).startswith("string"):
                m[c] = m[c] | df[c].astype("string").str.strip().eq("").fillna(False)
    return m


def etiqueta(col: str) -> str:
    clave = normalizar_nombre(col)
    return ETIQUETAS.get(clave, col.replace("_", " ").title())


def tabla_patrones(m: pd.DataFrame):
    n_falt = m.sum().sort_values(kind="stable")  # de menos a más faltantes
    con_na = [c for c in n_falt.index if n_falt[c] > 0]
    sin_na = [c for c in n_falt.index if n_falt[c] == 0]

    if not con_na:
        raise ValueError("No se encontraron datos faltantes en ninguna columna.")

    pat = m[con_na].value_counts().rename("frecuencia").reset_index()
    pat["n_variables_faltantes"] = pat[con_na].sum(axis=1)
    pat = pat.sort_values(
        ["frecuencia", "n_variables_faltantes"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    return pat, con_na, sin_na, n_falt


# ------------------------------------------------------------------
# FIGURA
# ------------------------------------------------------------------
def dibujar(pat, con_na, sin_na, n_falt, ruta_png):
    cols = [f"Otras sin faltantes ({len(sin_na)})"] + [etiqueta(c) for c in con_na]
    faltan_col = np.r_[0, [int(n_falt[c]) for c in con_na]]

    mat = np.zeros((len(pat), len(cols)), dtype=bool)
    mat[:, 1:] = pat[con_na].to_numpy(dtype=bool)

    n_f, n_c = mat.shape
    fig, ax = plt.subplots(figsize=(max(8.5, 1.8 * n_c + 2.5), max(4.8, 0.72 * n_f + 2.2)))

    for i in range(n_f):
        for j in range(n_c):
            ax.add_patch(
                plt.Rectangle(
                    (j, n_f - 1 - i),
                    1,
                    1,
                    facecolor=COLOR_FALTANTE if mat[i, j] else COLOR_PRESENTE,
                    edgecolor="white",
                    linewidth=2.2,
                )
            )

    ax.set_xlim(0, n_c)
    ax.set_ylim(0, n_f)
    ax.set_aspect("auto")
    for s in ax.spines.values():
        s.set_visible(False)

    # Etiquetas superiores (rotadas 45°; en rojo si la variable tiene faltantes)
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(n_c) + 0.5)
    ax.set_xticklabels(cols, rotation=45, ha="left", fontsize=8.5, fontweight="bold")
    for lab, k in zip(ax.get_xticklabels(), faltan_col):
        lab.set_color("#C53030" if k > 0 else "#4A5568")
    ax.tick_params(axis="x", length=4, color="#4A5568")

    # Frecuencia (izquierda) y n.º de variables faltantes (derecha)
    ax.set_yticks(np.arange(n_f) + 0.5)
    ax.set_yticklabels(
        [f"{int(v):,}" for v in pat["frecuencia"][::-1]],
        fontsize=8.5,
        fontweight="bold",
        color="#1A202C",
    )
    ax.tick_params(axis="y", length=0, pad=10)

    ax2 = ax.twinx()
    ax2.set_ylim(0, n_f)
    ax2.set_yticks(np.arange(n_f) + 0.5)
    ax2.set_yticklabels(
        [str(int(v)) for v in pat["n_variables_faltantes"][::-1]],
        fontsize=8.5,
        fontweight="bold",
        color="#1A202C",
    )
    ax2.tick_params(axis="y", length=0, pad=10)
    for s in ax2.spines.values():
        s.set_visible(False)

    # Totales de faltantes por columna (abajo)
    for j, k in enumerate(faltan_col):
        ax.text(
            j + 0.5,
            -0.14,
            f"{int(k):,}",
            ha="center",
            va="top",
            fontsize=8.5,
            fontweight="bold",
            color="#C53030" if k > 0 else "#4A5568",
            transform=ax.transData,
            clip_on=False,
        )

    ax.set_xlabel(
        f"Patrones de Datos Faltantes: Variable '{NOMBRE_SEGMENTO}'",
        fontsize=9.5,
        fontweight="bold",
        labelpad=24,
        color="#1A202C",
    )
    ax.xaxis.set_label_position("bottom")
    ax.set_ylabel("Frecuencia de registros", fontsize=9.5, fontweight="bold", labelpad=14)
    ax2.set_ylabel(
        "Cantidad de variables faltantes",
        fontsize=9.5,
        fontweight="bold",
        labelpad=18,
        rotation=270,
    )

    # Leyenda superior con recuadro fino
    legend = fig.legend(
        handles=[
            Patch(facecolor=COLOR_PRESENTE, edgecolor="#718096", linewidth=0.8, label="Dato Presente"),
            Patch(facecolor=COLOR_FALTANTE, edgecolor="none", label="Dato Faltante"),
        ],
        loc="upper center",
        ncol=2,
        frameon=True,
        edgecolor="#CBD5E0",
        facecolor="white",
        fontsize=8.5,
        bbox_to_anchor=(0.5, 1.01),
    )
    legend.get_frame().set_linewidth(0.8)

    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.93))
    fig.savefig(ruta_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------
if __name__ == "__main__":
    SALIDA.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RUTA_CSV, low_memory=False)
    m = mascara_faltantes(df)
    pat, con_na, sin_na, n_falt = tabla_patrones(m)

    resumen = pd.DataFrame(
        {
            "variable": n_falt.index,
            "n_faltantes": n_falt.to_numpy(),
            "pct_faltantes": (100 * n_falt / len(df)).round(3).to_numpy(),
        }
    )
    resumen.to_csv(SALIDA / "resumen_faltantes_por_variable.csv", index=False)
    pat.to_csv(SALIDA / "tabla_patrones_faltantes.csv", index=False)
    dibujar(pat, con_na, sin_na, n_falt, SALIDA / "figura_patrones_faltantes.png")

    print(f"Registros: {len(df):,} | variables: {df.shape[1]} | con faltantes: {len(con_na)}")
    print("\nPatrones (frecuencia, n.º de variables faltantes):")
    print(pat.to_string(index=False))
    print("\nFaltantes por variable:")
    print(resumen[resumen.n_faltantes > 0].to_string(index=False))
    print("\nSalidas en:", SALIDA.resolve())