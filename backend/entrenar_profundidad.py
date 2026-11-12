"""
entrenar_profundidad.py — Entrena el modelo de profundidad del agua subterranea
(Gradient Boosting + intervalo de error) y genera las tablas y figuras de la memoria.

Ejecutar desde la carpeta backend/:
    python entrenar_profundidad.py
    python entrenar_profundidad.py --sin-gee     # no recalcula la elevacion con GEE
    python entrenar_profundidad.py --solo-figuras   # rehace las figuras sin reentrenar

Para repetir el entrenamiento con exactamente los mismos datos (sin volver a llamar a GEE):
    python entrenar_profundidad.py --csv data/pozos_profundidad_v3.csv --sin-gee

Pasos:
  1. Lee data/pozos_andalucia_entrenamiento_filtrado.csv (pozos del IGME).
  2. Recalcula la elevacion de cada pozo con SRTM 30 m en Google Earth Engine:
     la MISMA fuente que usa la app al predecir (sin desajuste entrenamiento/produccion).
  3. Guarda data/pozos_profundidad_v3.csv (el CSV que puede subir el administrador).
  4. Entrena y valida con services/modelo_ia.entrenar():
     -> modelo/modelo_agua.pkl y modelo/metricas.json (los usa la app).
  5. Figuras en modelo/: fig_profundidad_eda.png, fig_profundidad_validacion.png,
     fig_profundidad_intervalo.png
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from services import modelo_ia  # noqa: E402

CSV_ENTRADA = "data/pozos_andalucia_entrenamiento_filtrado.csv"
CSV_SALIDA = "data/pozos_profundidad_v3.csv"
COLUMNAS = ["lat", "lon", "provincia", "profundidad_m", "elevacion", "geologia_score"]
DETALLE_PATH = modelo_ia.MODEL_DIR / "detalle_validacion.npz"


def _coma(valor, decimales=2):
    """Número con coma decimal, como en la memoria."""
    return f"{valor:.{decimales}f}".replace(".", ",")


def guardar_detalle(detalle, ruta=DETALLE_PATH):
    """Guarda las predicciones de validación para poder rehacer las figuras sin reentrenar."""
    planos = {}
    for clave, valor in detalle.items():
        if isinstance(valor, dict):
            for nombre, arr in valor.items():
                planos[f"pred::{nombre}"] = np.asarray(arr)
        else:
            planos[clave] = np.asarray(valor)
    np.savez(ruta, **planos)


def cargar_detalle(ruta=DETALLE_PATH):
    with np.load(ruta, allow_pickle=False) as datos:
        detalle = {"pred": {}}
        for clave in datos.files:
            if clave.startswith("pred::"):
                detalle["pred"][clave[len("pred::"):]] = datos[clave]
            else:
                detalle[clave] = datos[clave]
    return detalle


def preparar_csv(csv_entrada, csv_salida, usar_gee=True):
    df = pd.read_csv(csv_entrada)
    faltan = [c for c in ["lat", "lon", "profundidad_m", "geologia_score"] if c not in df.columns]
    if faltan:
        sys.exit(f"Faltan columnas en {csv_entrada}: {faltan}")
    df = df.dropna(subset=["lat", "lon", "profundidad_m", "geologia_score"])
    df = df[(df["profundidad_m"] > 0) & (df["profundidad_m"] <= modelo_ia.PROFUNDIDAD_MAX_M)].copy()
    print(f"Pozos con profundidad (0-{modelo_ia.PROFUNDIDAD_MAX_M:.0f} m) y geologia: {len(df)}")

    if usar_gee:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            from services.gee_service import init_gee
            if not init_gee():
                raise RuntimeError("init_gee() devolvio False")
            print("Calculando la elevacion SRTM 30 m de cada pozo en Google Earth Engine...")
            elev = modelo_ia.elevacion_srtm_gee(df["lat"].values, df["lon"].values)
            print(f"  Elevacion obtenida para {np.isfinite(elev).sum()} de {len(df)} pozos")
            if "elevacion" in df.columns:
                comunes = np.isfinite(elev) & df["elevacion"].notna().values
                if comunes.any():
                    dif = np.abs(elev[comunes] - df["elevacion"].values[comunes])
                    print(f"  Diferencia con la elevacion del CSV: mediana {np.median(dif):.1f} m")
            df["elevacion"] = elev
        except Exception as e:
            print(f"[AVISO] No se pudo usar GEE ({e}). Se usa la columna 'elevacion' del CSV.")
            if "elevacion" not in df.columns:
                sys.exit("El CSV no tiene columna 'elevacion' y GEE no esta disponible.")
    df = df.dropna(subset=["elevacion"])
    columnas = [c for c in COLUMNAS if c in df.columns]
    df[columnas].to_csv(csv_salida, index=False)
    print(f"Guardado: {csv_salida} ({len(df)} pozos)")
    return csv_salida


def dibujar_errores(ax, comp):
    """Barras de MAE y error mediano por modelo, con la leyenda en un hueco libre."""
    nombres = list(comp.keys())
    x = np.arange(len(nombres))
    mae = [comp[n]["mae_m"] for n in nombres]
    mediano = [comp[n]["mediana_error_m"] for n in nombres]
    ax.bar(x - 0.2, mae, 0.4, label="Error medio (MAE)")
    ax.bar(x + 0.2, mediano, 0.4, label="Error mediano")
    ax.set_xticks(x, [n.replace(" (referencia)", "\n(referencia)") for n in nombres], fontsize=8)
    ax.set_ylabel("Metros")
    ax.set_title("Error en validación espacial (bloques de 10 km)")
    # Se deja libre la franja superior y la leyenda va ahí, en una sola fila: no tapa ninguna barra.
    ax.set_ylim(0, max(mae + mediano) * 1.3)
    ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=9)


def figuras(metricas, detalle, salida: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib no instalado (pip install matplotlib): no se generan figuras.")
        return

    y = detalle["y"]
    # 1. Analisis exploratorio: distribucion de la profundidad y transformacion logaritmica
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
    ejes[0].hist(y, bins=40, color="#3b82f6")
    ejes[0].set_title(f"Profundidad de los pozos (asimetría = {_coma(pd.Series(y).skew())})")
    ejes[0].set_xlabel("Profundidad (m)")
    ejes[0].set_ylabel("N.º de pozos")
    ejes[1].hist(np.log1p(y), bins=40, color="#10b981")
    ejes[1].set_title(f"log(1 + profundidad) (asimetría = {_coma(pd.Series(np.log1p(y)).skew())})")
    ejes[1].set_xlabel("log(1 + m)")
    fig.tight_layout()
    fig.savefig(salida / "fig_profundidad_eda.png", dpi=200)
    plt.close(fig)

    # 2. Validacion: error de cada modelo y prediccion frente a valor real
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.8))
    dibujar_errores(ejes[0], metricas["comparacion"])
    p = detalle["pred"][modelo_ia.MODELO_GB]
    ejes[1].scatter(y, p, s=6, alpha=0.35)
    ejes[1].plot([0.5, 200], [0.5, 200], "k--", linewidth=1, label="Predicción perfecta")
    ejes[1].set_xscale("log")
    ejes[1].set_yscale("log")
    ejes[1].set_xlabel("Profundidad real (m)")
    ejes[1].set_ylabel("Profundidad predicha (m)")
    ejes[1].set_title("Gradient Boosting: predicho frente a real (fuera de pliegue)")
    ejes[1].legend()
    fig.tight_layout()
    fig.savefig(salida / "fig_profundidad_validacion.png", dpi=200)
    plt.close(fig)

    # 3. Intervalo de error: ejemplos y cobertura por distancia al pozo mas cercano
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(y), size=min(80, len(y)), replace=False))
    idx = idx[np.argsort(p[idx])]
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    pos = np.arange(len(idx))
    ejes[0].vlines(pos, detalle["lo"][idx], detalle["hi"][idx], color="#94a3b8", label="Intervalo del 80 %")
    ejes[0].scatter(pos, p[idx], s=10, color="#1d4ed8", label="Predicción", zorder=3)
    colores = np.where(detalle["dentro"][idx], "#16a34a", "#dc2626")
    ejes[0].scatter(pos, y[idx], s=14, c=colores, marker="x", label="Real (verde = dentro)", zorder=4)
    ejes[0].set_yscale("log")
    ejes[0].set_xlabel("Pozos de validación (ordenados por predicción)")
    ejes[0].set_ylabel("Profundidad (m)")
    ejes[0].set_title("Intervalos de predicción en pozos no vistos")
    ejes[0].legend(fontsize=8)
    tramos = metricas["intervalo"]["por_distancia"]
    etiquetas = [f"{t['tramo']}\n(n={t['n_pozos']})" for t in tramos]
    cob = [(t["cobertura"] or 0) * 100 for t in tramos]
    ejes[1].bar(etiquetas, cob, color="#3b82f6")
    ejes[1].axhline(metricas["intervalo"]["confianza"] * 100, color="k", linestyle="--", label="Objetivo")
    for i, t in enumerate(tramos):
        if t["anchura_mediana_m"] is not None:
            ejes[1].text(i, 3, f"anchura\n{t['anchura_mediana_m']:.0f} m", ha="center", color="white", fontsize=8)
    ejes[1].set_ylim(0, 100)
    ejes[1].set_ylabel("Cobertura real (%)")
    ejes[1].set_xlabel("Distancia al pozo conocido más cercano")
    ejes[1].set_title("Cobertura del intervalo por tramo")
    ejes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(salida / "fig_profundidad_intervalo.png", dpi=200)
    plt.close(fig)
    print(f"Figuras en {salida}: fig_profundidad_eda.png, fig_profundidad_validacion.png, fig_profundidad_intervalo.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=CSV_ENTRADA)
    parser.add_argument("--sin-gee", action="store_true", help="no recalcular la elevacion con GEE")
    parser.add_argument("--solo-figuras", action="store_true",
                        help="rehacer las figuras con los resultados guardados, sin reentrenar")
    args = parser.parse_args()

    if args.solo_figuras:
        if not (modelo_ia.METRICS_PATH.exists() and DETALLE_PATH.exists()):
            sys.exit("Faltan los resultados guardados. Entrena una vez con:\n"
                     "    python entrenar_profundidad.py --csv data/pozos_profundidad_v3.csv --sin-gee")
        metricas = json.loads(modelo_ia.METRICS_PATH.read_text(encoding="utf-8"))
        figuras(metricas, cargar_detalle(), modelo_ia.MODEL_DIR)
        return

    anterior = None
    if modelo_ia.METRICS_PATH.exists():
        anterior = json.loads(modelo_ia.METRICS_PATH.read_text(encoding="utf-8"))

    csv_modelo = preparar_csv(args.csv, CSV_SALIDA, usar_gee=not args.sin_gee)
    metricas, detalle = modelo_ia.entrenar(csv_modelo, devolver_detalle=True)
    guardar_detalle(detalle)

    if anterior and anterior.get("comparacion"):
        claves = [
            ("pozos", lambda m: m["n_pozos_entrenamiento"]),
            ("MAE del Gradient Boosting", lambda m: m["comparacion"][modelo_ia.MODELO_GB]["mae_m"]),
            ("R2 global", lambda m: m["metricas_cv"]["r2_global"]),
            ("cobertura del intervalo", lambda m: m["intervalo"]["cobertura"]),
        ]
        distintas = []
        for nombre, f in claves:
            try:
                if f(anterior) != f(metricas):
                    distintas.append(f"{nombre}: antes {f(anterior)}, ahora {f(metricas)}")
            except (KeyError, TypeError):
                distintas.append(f"{nombre}: no se pudo comparar")
        if distintas:
            print("\n[OJO] Los resultados han cambiado respecto al metricas.json anterior:")
            for d in distintas:
                print("   - " + d)
        else:
            print("\nResultados idénticos a los del metricas.json anterior.")

    print("\n" + "=" * 78)
    print("RESULTADOS PARA LA MEMORIA")
    print("=" * 78)
    print(f"Pozos: {metricas['n_pozos_entrenamiento']} | Hiperparametros: {metricas['mejores_hiperparametros']}")
    print(f"Validacion: {metricas['metricas_cv']['tipo_validacion']}\n")
    filas = [{"Modelo": n, **m} for n, m in metricas["comparacion"].items()]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    tabla = pd.DataFrame(filas)
    print(tabla.to_string(index=False))
    salida = modelo_ia.MODEL_DIR
    tabla.to_csv(salida / "tabla_profundidad_bloques.csv", index=False, encoding="utf-8-sig")
    if metricas["escenario_provincia_no_vista"]:
        esc = dict(metricas["escenario_provincia_no_vista"])
        n_prov = esc.pop("n_provincias")
        tabla_p = pd.DataFrame([{"Modelo": n, **m} for n, m in esc.items()])
        print(f"\nEscenario exigente (provincia entera sin pozos, {n_prov} provincias):")
        print(tabla_p.to_string(index=False))
        tabla_p.to_csv(salida / "tabla_profundidad_provincia.csv", index=False, encoding="utf-8-sig")
    iv = metricas["intervalo"]
    print(f"\nIntervalo del {iv['confianza']:.0%}: cobertura real {iv['cobertura']:.1%}, "
          f"anchura mediana {iv['anchura_mediana_m']} m")
    for t in iv["por_distancia"]:
        print(f"  {t['tramo']:<8} n={t['n_pozos']:<5} cobertura={t['cobertura']} anchura mediana={t['anchura_mediana_m']} m")
    print("\nImportancia de variables:")
    for k, v in metricas["importancia_variables"].items():
        print(f"  {k:<48} {v:.3f}")
    print("\n" + metricas["interpretacion"]["descripcion"])
    print(metricas["interpretacion"]["r2_texto"])
    print(metricas["interpretacion"]["intervalo_texto"])
    figuras(metricas, detalle, salida)


if __name__ == "__main__":
    main()
