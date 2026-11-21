import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from pathlib import Path

# =============================================================================
# Rutas (ajusta DATA_DIR a tu carpeta)
# =============================================================================
DATA_DIR = Path("../data/matrices")
F_XGB = DATA_DIR / "xgb_optuna_sow_siow_test_min80.csv"   # fuente principal del gráfico
F_REAL = DATA_DIR / "sow_siow_test_min80.csv"              # solo para validar
F_IDS = DATA_DIR / "ids_test_min80.csv"                    # solo para validar
OUT_DIR = Path(".")

COLORES = {
    "azul_oscuro": "#19375A", "azul": "#67B7E6", "azul_claro": "#DDF0FA",
    "rojo": "#F17C82", "rojo_claro": "#FCE5E5",
    "verde": "#88CF80", "verde_claro": "#E3F4DF",
    "amarillo": "#FFD36E", "amarillo_claro": "#FFF3CE",
    "gris": "#BFC4CB", "gris_claro": "#F0F1F3",
    "texto": "#182536", "blanco": "#FFFFFF",
}

# =============================================================================
# Cargar datos y validar contra los otros archivos
# =============================================================================
df = pd.read_csv(F_XGB)
cols = ["response_id", "SoW_real", "SoW_pred", "Siow_real", "Siow_pred"]
faltan = [c for c in cols if c not in df.columns]
assert not faltan, f"Faltan columnas en el CSV: {faltan}"

df = df.dropna(subset=["SoW_real", "Siow_real"]).reset_index(drop=True)
assert df["response_id"].is_unique, "response_id duplicados"

if F_REAL.exists():
    real = pd.read_csv(F_REAL)
    m = df.merge(real, on="response_id", how="outer", indicator=True)
    assert (m["_merge"] == "both").all(), "Los response_id no coinciden con sow_siow_test_min80.csv"
    assert np.allclose(m["SoW_real"], m["sow"]) and np.allclose(m["Siow_real"], m["siow"]), \
        "SoW/SioW reales del XGB difieren de sow_siow_test_min80.csv"
    print("OK: SoW/SioW reales coinciden con sow_siow_test_min80.csv")
if F_IDS.exists():
    ids = pd.read_csv(F_IDS)
    assert set(ids["response_id"]) == set(df["response_id"]), "Los IDs no coinciden con ids_test_min80.csv"
    print("OK: IDs coinciden con ids_test_min80.csv")

sow = df["SoW_real"].clip(0, 1).values
siow = df["Siow_real"].values.astype(float)
n_total = len(df)

# Rango de SoW_pred (para el texto del panel)
pred_min, pred_max = df["SoW_pred"].min() * 100, df["SoW_pred"].max() * 100

# =============================================================================
# Bins
# =============================================================================
sow_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0001]
sow_labels = ["0–20 %", "20–40 %", "40–60 %", "60–80 %", "80–100 %"]
sow_idx = np.clip(np.digitize(sow, sow_bins) - 1, 0, 4)

q = np.quantile(siow, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
for i in range(1, len(q)):
    if q[i] <= q[i - 1]:
        q[i] = q[i - 1] + 1e-9
siow_idx = np.clip(np.digitize(siow, q[1:-1], right=True), 0, 4)

fmt = lambda v: f"{v:.0f}" if abs(v - round(v)) < 1e-6 else f"{v:.1f}"
siow_labels = ["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto"]
siow_rangos = [f"[{fmt(q[0])}, {fmt(q[1])}]"] + [f"({fmt(q[k])}, {fmt(q[k+1])}]" for k in range(1, 5)]
siow_ticklabels = [f"{l}\n{r}" for l, r in zip(siow_labels, siow_rangos)]

# =============================================================================
# Conteos, porcentajes y gasto no capturado por celda
# =============================================================================
counts = np.zeros((5, 5), dtype=int)
uncaptured = np.zeros((5, 5))
for k in range(n_total):
    r, c = siow_idx[k], sow_idx[k]
    counts[r, c] += 1
    uncaptured[r, c] += siow[k] * (1.0 - sow[k])

pct = 100.0 * counts / n_total
pct_unc = 100.0 * uncaptured / uncaptured.sum()

print("\n=== Clientes por celda (filas = SioW, columnas = SoW) ===")
print(pd.DataFrame(counts, index=siow_labels, columns=sow_labels))
print("\n=== % de clientes por celda ===")
print(pd.DataFrame(np.round(pct, 1), index=siow_labels, columns=sow_labels))
print("\nCortes de quintiles SioW:", np.round(q, 2))
print("Clientes por quintil SioW:", counts.sum(axis=1).tolist())

# =============================================================================
# Categorías de color
# =============================================================================
categorias = [
    ["gris", "gris", "amarillo", "azul", "azul"],
    ["gris", "amarillo", "amarillo", "azul", "azul"],
    ["rojo", "rojo", "amarillo", "amarillo", "verde"],
    ["rojo", "rojo", "amarillo", "verde", "verde"],
    ["rojo", "rojo", "amarillo", "verde", "verde"],
]

# =============================================================================
# Tablas de datos de la gráfica
# =============================================================================
filas = []
for i in range(5):
    for j in range(5):
        filas.append({
            "SioW_segmento": siow_labels[i], "SioW_rango": siow_rangos[i],
            "SoW_segmento": sow_labels[j], "categoria": categorias[i][j],
            "n_clientes": counts[i, j], "pct_clientes": round(pct[i, j], 2),
            "gasto_no_capturado": round(uncaptured[i, j], 2),
            "pct_gasto_no_capturado": round(pct_unc[i, j], 2),
        })
tabla = pd.DataFrame(filas)
tabla.to_csv(OUT_DIR / "matriz_sow_siow_celdas.csv", index=False, encoding="utf-8-sig")

df_out = df.copy()
df_out["SoW_real"] = sow
df_out["SioW_quintil"] = [siow_labels[k] for k in siow_idx]
df_out["SoW_segmento"] = [sow_labels[k] for k in sow_idx]
df_out["categoria"] = [categorias[r][c] for r, c in zip(siow_idx, sow_idx)]
df_out["gasto_no_capturado"] = siow * (1 - sow)
df_out.to_csv(OUT_DIR / "matriz_sow_siow_clientes.csv", index=False, encoding="utf-8-sig")

resumen = tabla.groupby("categoria")[["n_clientes", "pct_clientes", "gasto_no_capturado", "pct_gasto_no_capturado"]].sum().round(2)
print("\n=== Resumen por categoría ===")
print(resumen)

# =============================================================================
# Gráfico
# =============================================================================
max_pct = pct.max() if pct.max() > 0 else 1

fig = plt.figure(figsize=(16, 9), facecolor="white")
ax = fig.add_axes([0.125, 0.215, 0.445, 0.665])
ax.set(xlim=(0, 5), ylim=(0, 5), aspect="equal")

for i in range(5):
    for j in range(5):
        cat = categorias[i][j]
        ax.add_patch(Rectangle((j, i), 1, 1, facecolor=COLORES[cat + "_claro"],
                               edgecolor="#C9D2DB", linewidth=0.9))
        if counts[i, j] > 0:
            radio = 0.12 + 0.30 * np.sqrt(pct[i, j] / max_pct)
            ax.add_patch(Circle((j + 0.5, i + 0.5), radio, facecolor=COLORES[cat],
                                edgecolor=COLORES["azul_oscuro"], linewidth=1.0, alpha=0.88))
            ax.text(j + 0.5, i + 0.5, f"{pct[i, j]:.1f}%\n(n={counts[i, j]})",
                    ha="center", va="center",
                    fontsize=8 if pct[i, j] < 8 else 9, fontweight="bold",
                    color=COLORES["azul_oscuro"] if cat in ("amarillo", "verde", "gris") else "white")

ax.set_xticks(np.arange(5) + 0.5, labels=sow_labels, fontsize=10)
ax.set_yticks(np.arange(5) + 0.5, labels=siow_ticklabels, fontsize=9.5)
ax.tick_params(axis="both", length=0, pad=9)
for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_xlabel("Share of Wallet (SoW) · participación Samsung (real)",
              fontsize=12, fontweight="bold", labelpad=15, color=COLORES["texto"])
ax.set_ylabel("Size of Wallet (SioW) · tamaño de cartera (real, quintiles)",
              fontsize=12, fontweight="bold", labelpad=19, color=COLORES["texto"])

def etiqueta(x, y, titulo, detalle, color):
    fig.text(x, y, titulo + "\n" + detalle, ha="center", va="center",
             fontsize=10, fontweight="bold", color=COLORES["azul_oscuro"],
             bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=color, lw=1.4))

def cat_stats(cat):
    s = tabla[tabla.categoria == cat]
    return s.n_clientes.sum(), s.pct_clientes.sum()

n_r, p_r = cat_stats("rojo"); n_v, p_v = cat_stats("verde")
n_g, p_g = cat_stats("gris"); n_a, p_a = cat_stats("azul")
etiqueta(0.215, 0.934, "ALTO POTENCIAL", f"SioW alto · SoW bajo · {p_r:.1f}% ({n_r})", COLORES["rojo"])
etiqueta(0.490, 0.934, "CLIENTES ESTRATÉGICOS", f"SioW alto · SoW alto · {p_v:.1f}% ({n_v})", COLORES["verde"])
etiqueta(0.215, 0.095, "BAJA PRIORIDAD", f"SioW bajo · SoW bajo · {p_g:.1f}% ({n_g})", COLORES["gris"])
etiqueta(0.490, 0.095, "ALTA PARTICIPACIÓN", f"SioW bajo · SoW alto · {p_a:.1f}% ({n_a})", COLORES["azul"])

def panel(pos, titulo):
    p = fig.add_axes(pos)
    p.set_xlim(0, 1); p.set_ylim(0, 1); p.axis("off")
    p.add_patch(FancyBboxPatch((0.01, 0.01), 0.98, 0.98, boxstyle="round,pad=0.006",
                               fc="white", ec="#C4D0DB", lw=1.3))
    p.add_patch(Rectangle((0.018, 0.79), 0.964, 0.185, fc=COLORES["azul_oscuro"], ec="none"))
    p.text(0.055, 0.883, titulo, color="white", fontsize=12.5, fontweight="bold", va="center")
    return p

gris_t = "#566475"
p1 = panel([0.625, 0.635, 0.355, 0.255], "Conceptos clave")
p1.text(0.065, 0.67, "SoW = gasto Samsung / cartera tech total", fontsize=10.6, va="center")
p1.text(0.065, 0.52, "SioW = tamaño de la cartera del cliente", fontsize=10.6, va="center")
p1.text(0.065, 0.35, "Gasto no capturado = SioW × (1 − SoW)", fontsize=11, fontweight="bold", va="center")
p1.text(0.065, 0.17, f"Datos: test min80 · n = {n_total} clientes", fontsize=9.6, va="center", color=gris_t)

p2 = panel([0.625, 0.375, 0.355, 0.235], "Tamaño de las burbujas")
p2.text(0.065, 0.66, "Área ∝ % de clientes en la celda", fontsize=10.5, va="center")
p2.text(0.065, 0.48, "Dentro: % del test y número de clientes (n)", fontsize=10.5, va="center")
p2.text(0.065, 0.28, f"SoW_pred XGB no se usa (rango {pred_min:.1f}–{pred_max:.1f} %)", fontsize=9.2, va="center", color=gris_t)
p2.text(0.065, 0.14, "(sin discriminación entre segmentos)", fontsize=9.0, va="center", color=gris_t)

p3 = panel([0.625, 0.115, 0.355, 0.235], "Lectura de la matriz")
p3.text(0.065, 0.62, f"Rojo: conquistar · {pct_unc[np.array(categorias)=='rojo'].sum():.0f}% del gasto no capturado", fontsize=9.6, va="center")
p3.text(0.065, 0.45, f"Verde: retener / maximizar · {pct_unc[np.array(categorias)=='verde'].sum():.0f}% del gasto no capturado", fontsize=9.6, va="center")
p3.text(0.065, 0.28, "Azul: defender cuota en carteras pequeñas", fontsize=9.6, va="center")
p3.text(0.065, 0.12, "Gris: baja prioridad relativa", fontsize=9.6, va="center")

out_path = OUT_DIR / "matriz_burbujas_sow_siow_real.png"
fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
print(f"\nGráfico guardado en: {out_path}")
plt.show()
