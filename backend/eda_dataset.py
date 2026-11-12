"""
eda_dataset.py — Analisis exploratorio (EDA) de los dos datasets del TFM.

  A. Pozos del IGME (modelo de profundidad):
     - estadisticos descriptivos y valores ausentes por columna
     - distribucion de la profundidad: asimetria, valores > 200 m, transformacion log
     - profundidad por provincia y por clase geologica
     - relacion con la elevacion y con la profundidad de los pozos vecinos (IDW)
     - autocorrelacion espacial: diferencia de profundidad entre pozos segun la
       distancia que los separa (justifica el IDW y la validacion espacial)
  B. Parches de la U-Net (si ya estan descargados):
     - proporcion de pixeles de agua (desequilibrio de clases)
     - firma espectral del agua frente al resto (justifica las bandas y el MNDWI)

Ejecutar desde backend/ (no necesita Google Earth Engine; tarda segundos):
    python eda_dataset.py
Genera en modelo/eda/: tablas .csv y figuras .png para el apartado 4.3 de la memoria.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))
from services.modelo_ia import PROFUNDIDAD_MAX_M, _distancia_m_vec, _vecinos  # noqa: E402

CANDIDATOS_BRUTO = ["data/pozos_andalucia_enriquecido.csv", "data/pozos_andalucia_listo.csv",
                    "data/pozos_andalucia_entrenamiento_filtrado.csv"]
CANDIDATOS_MODELO = ["data/pozos_profundidad_v3.csv", "data/pozos_andalucia_entrenamiento_filtrado.csv"]
PARCHES = BACKEND / "unet" / "parches_agua"
BANDAS = ["B2", "B3", "B4", "B8", "B11", "B12"]
NOMBRES_BANDAS = ["Azul (B2)", "Verde (B3)", "Rojo (B4)", "NIR (B8)", "SWIR1 (B11)", "SWIR2 (B12)"]
SALIDA = BACKEND / "modelo" / "eda"
SEMILLA = 42


def primero_que_exista(rutas):
    for r in rutas:
        if (BACKEND / r).exists():
            return BACKEND / r
    return None


def spearman(a, b):
    a, b = pd.Series(a), pd.Series(b)
    return float(a.rank().corr(b.rank()))


def variograma(lat, lon, valor, tramos_km=(0, 1, 2, 5, 10, 20, 50, 100), max_pozos=1500):
    """Mediana de |diferencia de log(profundidad)| entre pares de pozos por tramo de distancia."""
    rng = np.random.default_rng(SEMILLA)
    idx = rng.choice(len(lat), size=min(len(lat), max_pozos), replace=False)
    la, lo, v = lat[idx], lon[idx], valor[idx]
    difs = [[] for _ in range(len(tramos_km) - 1)]
    for i in range(len(idx) - 1):
        d = _distancia_m_vec(la[i], lo[i], la[i + 1:], lo[i + 1:]) / 1000
        dv = np.abs(v[i] - v[i + 1:])
        t = np.digitize(d, tramos_km) - 1
        for k in range(len(difs)):
            difs[k].append(dv[t == k])
    filas = []
    for k in range(len(difs)):
        x = np.concatenate(difs[k]) if difs[k] else np.array([])
        filas.append({"tramo_km": f"{tramos_km[k]}-{tramos_km[k + 1]}", "n_pares": int(len(x)),
                      "mediana_dif_log": round(float(np.median(x)), 3) if len(x) else None})
    return pd.DataFrame(filas)


def eda_pozos(csv_bruto, csv_modelo, plt):
    print(f"\n=== A. POZOS DEL IGME ===\nDataset bruto: {csv_bruto.name} | dataset del modelo: {csv_modelo.name}")
    bruto = pd.read_csv(csv_bruto)
    print(f"Filas: {len(bruto)} | columnas: {len(bruto.columns)}")

    # 1. Descriptivos y valores ausentes (Tema 3: datos ausentes)
    num = bruto.select_dtypes(include="number")
    desc = num.describe().T
    desc.insert(1, "ausentes_%", (num.isna().mean() * 100).round(1))
    desc["asimetria"] = num.skew().round(2)
    desc = desc.round(3)
    desc.to_csv(SALIDA / "tabla_eda_descriptiva.csv", encoding="utf-8-sig")
    print("\nEstadisticos descriptivos (columnas numericas):")
    print(desc[["count", "ausentes_%", "mean", "50%", "min", "max", "asimetria"]].to_string())

    ausentes = (bruto.isna().mean() * 100).sort_values(ascending=False)
    ausentes = ausentes[ausentes > 0]
    if len(ausentes):
        fig, eje = plt.subplots(figsize=(8, max(3, 0.35 * len(ausentes))))
        eje.barh(ausentes.index[::-1], ausentes.values[::-1], color="#f59e0b")
        eje.set_xlabel("% de filas sin valor")
        eje.set_title("Valores ausentes por columna (dataset bruto)")
        fig.tight_layout()
        fig.savefig(SALIDA / "fig_eda_ausentes.png", dpi=200)
        plt.close(fig)

    # 2. Distribucion de la profundidad y valores extremos
    prof_todos = bruto["profundidad_m"].dropna()
    prof_todos = prof_todos[prof_todos > 0]
    n_mayor = int((prof_todos > PROFUNDIDAD_MAX_M).sum())
    dup = int(bruto.duplicated(subset=["lat", "lon", "profundidad_m"]).sum())
    print(f"\nPozos con profundidad > 0: {len(prof_todos)} de {len(bruto)} ({len(prof_todos) / len(bruto) * 100:.1f} %)")
    print(f"Pozos con mas de {PROFUNDIDAD_MAX_M:.0f} m (se excluyen): {n_mayor} ({n_mayor / len(prof_todos) * 100:.1f} %)")
    print(f"Duplicados exactos (mismo punto y profundidad): {dup}")

    df = pd.read_csv(csv_modelo).dropna(subset=["lat", "lon", "profundidad_m", "elevacion", "geologia_score"])
    df = df[(df["profundidad_m"] > 0) & (df["profundidad_m"] <= PROFUNDIDAD_MAX_M)]
    df = df.drop_duplicates(subset=["lat", "lon", "profundidad_m"]).reset_index(drop=True)
    p = df["profundidad_m"].values
    print(f"Dataset del modelo: {len(df)} pozos | mediana {np.median(p):.1f} m | media {p.mean():.1f} m | "
          f"asimetria {pd.Series(p).skew():.2f} -> en log {pd.Series(np.log1p(p)).skew():.2f}")

    fig, ejes = plt.subplots(1, 3, figsize=(15, 4.2))
    ejes[0].hist(prof_todos.clip(upper=400), bins=60, color="#3b82f6")
    ejes[0].axvline(PROFUNDIDAD_MAX_M, color="red", linestyle="--", label=f"Corte {PROFUNDIDAD_MAX_M:.0f} m")
    ejes[0].set_title("Profundidad (todos los pozos, recortado a 400 m)")
    ejes[0].set_xlabel("m")
    ejes[0].legend()
    ejes[1].hist(np.log1p(p), bins=40, color="#10b981")
    ejes[1].set_title(f"log(1 + profundidad), asimetria {pd.Series(np.log1p(p)).skew():.2f}")
    ejes[1].set_xlabel("log(1 + m)")
    if "provincia" in df.columns:
        orden = df.groupby("provincia")["profundidad_m"].median().sort_values().index
        ejes[2].boxplot([df.loc[df.provincia == pr, "profundidad_m"] for pr in orden], showfliers=False)
        ejes[2].set_xticks(range(1, len(orden) + 1), orden, rotation=45, ha="right", fontsize=8)
        ejes[2].set_title("Profundidad por provincia (sin atipicos)")
        ejes[2].set_ylabel("m")
        tabla_prov = df.groupby("provincia")["profundidad_m"].agg(
            n_pozos="count", mediana_m="median", media_m="mean", p25_m=lambda s: s.quantile(.25),
            p75_m=lambda s: s.quantile(.75)).round(1).sort_values("mediana_m")
        tabla_prov.to_csv(SALIDA / "tabla_eda_provincias.csv", encoding="utf-8-sig")
        print("\nProfundidad por provincia:")
        print(tabla_prov.to_string())
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_eda_profundidad.png", dpi=200)
    plt.close(fig)

    # 3. Relaciones con las variables del modelo
    lat, lon = df["lat"].values, df["lon"].values
    idw, dmin, _ = _vecinos(lat, lon, lat, lon, p, excluir_propio=True)
    rho_elev = spearman(df["elevacion"], p)
    rho_idw = spearman(idw, p)
    rho_geo = spearman(df["geologia_score"], p)
    rho_dist = spearman(dmin, np.abs(np.log1p(idw) - np.log1p(p)))
    print(f"\nCorrelacion de Spearman con la profundidad: elevacion {rho_elev:.2f} | IDW de los 5 vecinos {rho_idw:.2f} | "
          f"geologia {rho_geo:.2f}")
    print(f"Distancia al vecino mas cercano frente al error del IDW: {rho_dist:.2f} "
          f"(positiva = cuanto mas lejos esta el vecino, mayor error)")
    print(f"Distancia al pozo mas cercano: mediana {np.median(dmin):.2f} km | p90 {np.percentile(dmin, 90):.2f} km")

    fig, ejes = plt.subplots(1, 3, figsize=(15, 4.2))
    ejes[0].scatter(df["elevacion"], p, s=5, alpha=0.3)
    ejes[0].set_yscale("log")
    ejes[0].set_xlabel("Elevacion (m)")
    ejes[0].set_ylabel("Profundidad (m)")
    ejes[0].set_title(f"Elevacion (Spearman = {rho_elev:.2f})")
    ejes[1].scatter(idw, p, s=5, alpha=0.3, color="#16a34a")
    ejes[1].plot([0.5, 200], [0.5, 200], "k--", linewidth=1)
    ejes[1].set_xscale("log")
    ejes[1].set_yscale("log")
    ejes[1].set_xlabel("IDW de los 5 pozos vecinos (m)")
    ejes[1].set_title(f"Pozos vecinos (Spearman = {rho_idw:.2f})")
    clases = sorted(df["geologia_score"].unique())
    etiquetas = {0.0: "Desfavorable", 0.5: "Moderada", 1.0: "Favorable"}
    ejes[2].boxplot([df.loc[df.geologia_score == c, "profundidad_m"] for c in clases], showfliers=False)
    ejes[2].set_xticks(range(1, len(clases) + 1),
                       [f"{etiquetas.get(c, c)}\n(n={int((df.geologia_score == c).sum())})" for c in clases])
    ejes[2].set_ylabel("Profundidad (m)")
    ejes[2].set_title(f"Geologia (Spearman = {rho_geo:.2f})")
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_eda_relaciones.png", dpi=200)
    plt.close(fig)

    # 4. Autocorrelacion espacial
    vg = variograma(lat, lon, np.log1p(p))
    vg.to_csv(SALIDA / "tabla_eda_variograma.csv", index=False, encoding="utf-8-sig")
    print("\nDiferencia tipica de log(profundidad) entre pozos segun su distancia:")
    print(vg.to_string(index=False))
    fig, ejes = plt.subplots(1, 2, figsize=(13, 5))
    mapa = ejes[0].scatter(lon, lat, c=np.log1p(p), s=6, cmap="viridis_r")
    cb = fig.colorbar(mapa, ax=ejes[0])
    marcas = [2, 5, 10, 20, 50, 100, 200]
    cb.set_ticks(np.log1p(marcas), labels=[f"{m} m" for m in marcas])
    ejes[0].set_xlabel("Longitud")
    ejes[0].set_ylabel("Latitud")
    ejes[0].set_title("Profundidad de los pozos del IGME")
    ejes[0].set_aspect(1 / np.cos(np.radians(lat.mean())))
    validos = vg.dropna()
    ejes[1].plot(range(len(validos)), validos["mediana_dif_log"], "o-", color="#dc2626")
    ejes[1].set_xticks(range(len(validos)), validos["tramo_km"])
    ejes[1].set_xlabel("Distancia entre pozos (km)")
    ejes[1].set_ylabel("Mediana de |diferencia de log(profundidad)|")
    ejes[1].set_title("Diferencia de profundidad segun la distancia entre pozos")
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_eda_espacial.png", dpi=200)
    plt.close(fig)


def eda_parches(plt):
    archivos = sorted(PARCHES.glob("parche_*.npz"))
    if not archivos:
        print(f"\n=== B. PARCHES DE LA U-NET ===\nNo hay parches en {PARCHES} (ejecuta antes unet/descargar_parches.py)")
        return
    X = np.stack([np.load(a)["imagen"] for a in archivos]).astype(np.float32)
    Y = np.stack([np.load(a)["mascara"] for a in archivos]).astype(bool)
    pct = Y.reshape(len(Y), -1).mean(axis=1) * 100
    print(f"\n=== B. PARCHES DE LA U-NET ===\nParches: {len(X)} | pixeles de agua: {Y.mean() * 100:.1f} % "
          f"| parches sin agua: {(pct == 0).mean() * 100:.0f} %")
    agua = X.transpose(1, 0, 2, 3)[:, Y]
    resto = X.transpose(1, 0, 2, 3)[:, ~Y]
    firma = pd.DataFrame({"banda": NOMBRES_BANDAS,
                          "agua_media": agua.mean(axis=1).round(4), "agua_desv": agua.std(axis=1).round(4),
                          "resto_media": resto.mean(axis=1).round(4), "resto_desv": resto.std(axis=1).round(4)})
    firma.to_csv(SALIDA / "tabla_eda_firma_espectral.csv", index=False, encoding="utf-8-sig")
    print("Firma espectral media (reflectancia):")
    print(firma.to_string(index=False))

    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(BANDAS))
    for nombre, datos, color in (("Agua", agua, "#2563eb"), ("Resto", resto, "#a16207")):
        ejes[0].errorbar(x, datos.mean(axis=1), yerr=datos.std(axis=1), marker="o", capsize=3, label=nombre, color=color)
    ejes[0].set_xticks(x, NOMBRES_BANDAS, rotation=30, ha="right", fontsize=8)
    ejes[0].set_ylabel("Reflectancia")
    ejes[0].set_title("Firma espectral: agua frente al resto (media y desviacion)")
    ejes[0].legend()
    ejes[1].hist(pct, bins=30, color="#2563eb")
    ejes[1].set_xlabel("% de pixeles de agua en el parche")
    ejes[1].set_ylabel("N.o de parches")
    ejes[1].set_title(f"Desequilibrio de clases: {Y.mean() * 100:.1f} % de pixeles son agua")
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_eda_unet.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-bruto", default=None)
    parser.add_argument("--csv-modelo", default=None)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SALIDA.mkdir(parents=True, exist_ok=True)
    csv_bruto = Path(args.csv_bruto) if args.csv_bruto else primero_que_exista(CANDIDATOS_BRUTO)
    csv_modelo = Path(args.csv_modelo) if args.csv_modelo else primero_que_exista(CANDIDATOS_MODELO)
    if csv_bruto is None or csv_modelo is None:
        sys.exit("No encuentro los CSV de pozos en data/. Indica las rutas con --csv-bruto y --csv-modelo.")
    eda_pozos(csv_bruto, csv_modelo, plt)
    eda_parches(plt)
    print(f"\nTablas y figuras en {SALIDA}")


if __name__ == "__main__":
    main()
