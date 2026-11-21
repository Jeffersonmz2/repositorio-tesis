"""
Cruce de clústeres K-Means (TRAIN) con SoW y SioW.

Alineación (verificada):
  clusters_train{TAG}.csv trae response_id sintético (train_00001...) en el MISMO orden que
  la matriz W usada en K-Means = ids_train{TAG}.csv. Se reemplaza por el response_id real
  por posición y luego se cruza por id con sow_siow_train{TAG}.csv.
"""
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu

# ------------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------------
TAG = "_min80"
RUTA_CLUSTERS = Path(f"resultados_kmeans_python{TAG}/clusters_train{TAG}.csv")
RUTA_CLIENTES_W = Path(f"resultados_kmeans_python{TAG}/clientes_W_train_con_cluster{TAG}.csv")  # para verificar
RUTA_IDS_TRAIN = Path(f"ids_train{TAG}.csv")
RUTA_RESPUESTAS = Path(f"sow_siow_train{TAG}.csv")          # response_id, sow, siow
RUTA_MATRICES = Path(f"matrices_data{TAG}.xlsx")            # hoja W_train, para verificar orden
VARIABLES = ["sow", "siow"]
SALIDA = Path(f"cruce_clusters_sow_siow{TAG}")
SALIDA.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# CARGA, ALINEACIÓN Y VERIFICACIÓN
# ------------------------------------------------------------------
def cargar_cruce():
    clusters = pd.read_csv(RUTA_CLUSTERS, dtype={"response_id": "string"})
    ids = pd.read_csv(RUTA_IDS_TRAIN, dtype={"response_id": "string"})
    resp = pd.read_csv(RUTA_RESPUESTAS, dtype={"response_id": "string"})

    if len(clusters) != len(ids):
        raise ValueError(f"clusters={len(clusters)} vs ids_train={len(ids)}: TAG distinto.")
    if not ids.response_id.is_unique:
        raise ValueError("ids_train tiene response_id duplicados")

    # Verificación fuerte del orden: W de la matriz == W usada para los clusters
    if RUTA_MATRICES.is_file() and RUTA_CLIENTES_W.is_file():
        W = pd.read_excel(RUTA_MATRICES, sheet_name="W_train")
        cw = pd.read_csv(RUTA_CLIENTES_W)
        cols = [c for c in cw.columns if c in W.columns and c.lower() != "intercept"]
        dif = np.abs(W[cols].to_numpy(float) - cw[cols].to_numpy(float)).max()
        if dif > 1e-9:
            raise ValueError(f"El orden de filas NO coincide con matrices_data (dif máx {dif:g}).")
        print(f"Verificación de orden OK: W de {RUTA_MATRICES.name} == W usada en clusters ({len(cols)} variables)")

    base = clusters[["cluster"]].copy()
    base.insert(0, "response_id", ids.response_id.to_numpy())      # id real por posición
    m = base.merge(resp[["response_id"] + VARIABLES], on="response_id", how="left", validate="one_to_one")
    if m[VARIABLES].isna().any().any():
        raise ValueError("Clientes sin SoW/SioW tras el cruce por response_id")
    return m


# ------------------------------------------------------------------
# ANÁLISIS
# ------------------------------------------------------------------
def resumen(m, var):
    g = m.groupby("cluster")[var]
    t = g.agg(n="size", media="mean", mediana="median", sd="std",
              q25=lambda s: s.quantile(.25), q75=lambda s: s.quantile(.75), maximo="max")
    t["pct_ceros"] = g.apply(lambda s: 100 * (s == 0).mean())
    t["media_si_mayor_0"] = g.apply(lambda s: s[s > 0].mean())
    total = pd.DataFrame([{
        "n": len(m), "media": m[var].mean(), "mediana": m[var].median(), "sd": m[var].std(),
        "q25": m[var].quantile(.25), "q75": m[var].quantile(.75), "maximo": m[var].max(),
        "pct_ceros": 100 * (m[var] == 0).mean(), "media_si_mayor_0": m.loc[m[var] > 0, var].mean(),
    }], index=["Total"])
    t.index = t.index.astype(str)
    return pd.concat([t, total]).rename_axis("cluster").reset_index()


def holm(p):
    p = np.asarray(p); orden = np.argsort(p); m = len(p); aj = np.empty(m); previo = 0
    for rango, i in enumerate(orden):
        previo = max(previo, (m - rango) * p[i]); aj[i] = min(1.0, previo)
    return aj


def pruebas(m, var):
    grupos = {c: g[var].to_numpy() for c, g in m.groupby("cluster")}
    h, p = kruskal(*grupos.values())
    n, k = len(m), len(grupos)
    glob = {"variable": var, "prueba": "Kruskal-Wallis (valores)", "estadistico": h, "gl": k - 1,
            "p_valor": p, "medida_efecto": "epsilon2", "tamano_efecto": (h - k + 1) / (n - k)}
    tabla = pd.crosstab(m["cluster"], m[var] > 0)
    chi, pc, gl, _ = chi2_contingency(tabla)
    cramer = np.sqrt(chi / (n * (min(tabla.shape) - 1)))
    glob2 = {"variable": var, "prueba": "Chi-cuadrado (>0 vs 0)", "estadistico": chi, "gl": gl,
             "p_valor": pc, "medida_efecto": "V de Cramer", "tamano_efecto": cramer}
    pares = []
    for a, b in combinations(grupos, 2):
        u, pu = mannwhitneyu(grupos[a], grupos[b], alternative="two-sided")
        pares.append({"variable": var, "cluster_a": a, "cluster_b": b, "U": u, "p_valor": pu})
    pares = pd.DataFrame(pares); pares["p_holm"] = holm(pares.p_valor)
    return pd.DataFrame([glob, glob2]), pares


def grafico(m, ruta):
    clusters = sorted(m.cluster.unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, var, tit in zip(axes, VARIABLES, ["SoW (share of wallet)", "SioW (size of wallet)"]):
        datos = [m.loc[m.cluster == c, var] for c in clusters]
        ax.boxplot(datos, tick_labels=[f"C{c}\n(n={len(d)})" for c, d in zip(clusters, datos)],
                   showmeans=True, showfliers=False)
        ax.set_title(f"{tit} por clúster (sin outliers; ▲ = media)")
        ax.set_ylabel(var); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(ruta, dpi=180); plt.close(fig)


if __name__ == "__main__":
    m = cargar_cruce()
    m.to_csv(SALIDA / f"cruce_clientes_clusters_sow_siow{TAG}.csv", index=False)
    glob_all, pares_all = [], []
    for var in VARIABLES:
        t = resumen(m, var); t.to_csv(SALIDA / f"resumen_{var}_por_cluster.csv", index=False)
        print(f"\n{var.upper()} por clúster:\n", t.round(4).to_string(index=False))
        g, p = pruebas(m, var); glob_all.append(g); pares_all.append(p)
    pd.concat(glob_all).to_csv(SALIDA / "pruebas_globales.csv", index=False)
    pd.concat(pares_all).to_csv(SALIDA / "comparaciones_por_pares_holm.csv", index=False)
    print("\nPruebas globales:\n", pd.concat(glob_all).round(4).to_string(index=False))
    print("\nPares (Mann-Whitney, Holm):\n", pd.concat(pares_all).round(4).to_string(index=False))
    grafico(m, SALIDA / "boxplots_sow_siow_por_cluster.png")
    print("\nSalidas en:", SALIDA.resolve())
