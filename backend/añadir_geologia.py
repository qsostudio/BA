"""
añadir_geologia.py
Lee pozos_igme_completos.csv y añade geologia_score real
consultando la API de Macrostrat para cada pozo.

Ejecutar desde la carpeta backend/:
    python añadir_geologia.py

Tarda unos 10-15 minutos para 500 pozos.
"""

import requests
import pandas as pd
import time
from pathlib import Path

CSV_ENTRADA = "data/pozos_igme_completos.csv"
CSV_SALIDA  = "data/pozos_igme_final.csv"
PAUSA       = 0.4   # segundos entre llamadas a Macrostrat

ROCAS_BUENAS = ["limestone", "carbonate", "dolomite", "sandstone", "gravel", "alluvium"]
ROCAS_MALAS  = ["granite", "schist", "slate", "gneiss", "basalt"]


def clasificar_geologia(lith: str) -> float:
    if not lith or pd.isna(lith):
        return 0.5
    lith_lower = str(lith).lower()
    if any(r in lith_lower for r in ROCAS_BUENAS):
        return 1.0
    elif any(r in lith_lower for r in ROCAS_MALAS):
        return 0.0
    return 0.5


def consultar_macrostrat(lat: float, lon: float) -> float:
    try:
        r = requests.get(
            "https://macrostrat.org/api/mobile/map_query",
            params={"lng": lon, "lat": lat, "z": 12},
            timeout=10
        )
        data     = r.json()
        geo_list = data.get("success", {}).get("data", {}).get("mapData", [])
        if geo_list:
            lith = geo_list[0].get("lith", "")
            return clasificar_geologia(lith)
    except Exception:
        pass
    return 0.5


def main():
    print("=" * 60)
    print("  AÑADIR GEOLOGÍA REAL — MACROSTRAT")
    print("=" * 60)

    if not Path(CSV_ENTRADA).exists():
        print(f"\n❌ No se encuentra {CSV_ENTRADA}")
        print("   Ejecuta primero extraer_features_gee.py")
        return

    df = pd.read_csv(CSV_ENTRADA)
    print(f"\n[DATOS] {len(df)} pozos cargados")
    print("[API]   Consultando Macrostrat para cada pozo...")
    print("        Esto tardará unos 10-15 minutos\n")

    scores   = []
    total    = len(df)
    errores  = 0

    for i, row in df.iterrows():
        lat = row["lat"]
        lon = row["lon"]

        score = consultar_macrostrat(lat, lon)
        scores.append(score)

        # Progreso cada 50 pozos
        n = i + 1
        if n % 50 == 0 or n == total:
            n_buena = scores.count(1.0)
            n_mala  = scores.count(0.0)
            n_media = scores.count(0.5)
            print(f"  {n}/{total} pozos procesados "
                  f"— Buena: {n_buena} | Media: {n_media} | Mala: {n_mala}")

        time.sleep(PAUSA)

    df["geologia_score"] = scores

    # Guardar
    Path("data").mkdir(exist_ok=True)
    df.to_csv(CSV_SALIDA, index=False)

    n_buena = scores.count(1.0)
    n_mala  = scores.count(0.0)
    n_media = scores.count(0.5)

    print(f"\n{'='*60}")
    print(f"  ✅ GEOLOGÍA AÑADIDA")
    print(f"{'='*60}")
    print(f"  Geología favorable (caliza, arenisca...): {n_buena} pozos")
    print(f"  Geología media:                          {n_media} pozos")
    print(f"  Geología desfavorable (granito, gneis...): {n_mala} pozos")
    print(f"\n  Fichero guardado: {CSV_SALIDA}")
    print(f"\n  SIGUIENTE PASO:")
    print(f"  Sube {CSV_SALIDA} desde el panel de entrenamiento de la app.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
