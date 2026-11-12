"""
descargar_pozos_nacional.py
Descarga pozos del Inventario de Puntos de Agua del IGME para TODA ESPAÑA
(o las provincias que elijas) directamente desde el servicio público
ArcGIS REST del IGME, sin depender del Excel manual.

Fuente: https://mapas.igme.es/gis/rest/services/BasesDatos/IGME_PuntosAgua/MapServer

Ejecutar desde backend/:
    python descargar_pozos_nacional.py

El servicio limita cada consulta a 1000 registros, así que el script
pagina automáticamente hasta descargar todos los puntos disponibles.
"""

import requests
import pandas as pd
import time
from pathlib import Path

# ── Configuración ────────────────────────────────────────────────
BASE_URL = "https://mapas.igme.es/gis/rest/services/BasesDatos/IGME_PuntosAgua/MapServer/1/query"
PAGE_SIZE = 1000          # máximo permitido por el servicio
PAUSA = 0.3               # segundos entre páginas, para no saturar el servidor
CSV_SALIDA = "data/pozos_igme_nacional.csv"

# Si quieres limitar a ciertas provincias, escribe sus nombres aquí.
# Déjalo como lista vacía [] para descargar TODA ESPAÑA.
PROVINCIAS_FILTRO = ["Huelva", "Sevilla", "Cádiz", "Córdoba", "Granada", "Málaga", "Almería", "Jaén"]


def construir_where():
    if not PROVINCIAS_FILTRO:
        return "1=1"
    provincias_sql = ",".join(f"'{p}'" for p in PROVINCIAS_FILTRO)
    return f"Provincia IN ({provincias_sql})"


def descargar_pagina(offset, where):
    params = {
        "where": where,
        "outFields": "*",
        "f": "geojson",
        "resultRecordCount": PAGE_SIZE,
        "resultOffset": offset,
        "outSR": "4326",  # latitud/longitud estándar
    }
    resp = requests.get(BASE_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def main():
    print("=" * 60)
    print("  DESCARGA NACIONAL DE POZOS — IGME (ArcGIS REST)")
    print("=" * 60)

    where = construir_where()
    ambito = ", ".join(PROVINCIAS_FILTRO) if PROVINCIAS_FILTRO else "TODA ESPAÑA"
    print(f"\n[ÁMBITO] {ambito}")
    print(f"[FILTRO SQL] {where}\n")

    todos_los_pozos = []
    offset = 0
    pagina = 1

    while True:
        print(f"  Página {pagina} (offset {offset})...", end=" ", flush=True)
        try:
            data = descargar_pagina(offset, where)
        except Exception as e:
            print(f"⚠️  Error: {e}")
            break

        features = data.get("features", [])
        n = len(features)
        print(f"✅ {n} pozos")

        if n == 0:
            break

        for feat in features:
            props = feat.get("properties", {})
            geom = feat.get("geometry", {})
            coords = geom.get("coordinates", [None, None])

            todos_los_pozos.append({
                "lon": coords[0],
                "lat": coords[1],
                "provincia": props.get("Provincia"),
                "municipio": props.get("Municipio"),
                "naturaleza": props.get("Naturaleza"),
                "utilizacion": props.get("Utilizacion") or props.get("Utilización"),
                "cuenca_hidrografica": props.get("Cuenca_Hidrografica"),
                "unidad_hidrogeologica": props.get("Unidad_Hidrogeologica"),
                "sistema_acuifero": props.get("Sistema_Acuifero"),
                "profundidad": props.get("Profundidad") or props.get("PROFUNDIDAD"),
                "cota": props.get("Cota") or props.get("COTA"),
            })

        if n < PAGE_SIZE:
            break  # última página

        offset += PAGE_SIZE
        pagina += 1
        time.sleep(PAUSA)

    if not todos_los_pozos:
        print("\n❌ No se descargó ningún pozo. Revisa la conexión o el filtro.")
        return

    df = pd.DataFrame(todos_los_pozos)
    df = df.dropna(subset=["lat", "lon"])

    Path("data").mkdir(exist_ok=True)
    df.to_csv(CSV_SALIDA, index=False)

    print(f"\n{'='*60}")
    print(f"  ✅ DESCARGA COMPLETADA")
    print(f"  Total pozos: {len(df)}")
    if "provincia" in df.columns:
        print(f"\n  Distribución por provincia (top 10):")
        print(df["provincia"].value_counts().head(10).to_string())
    print(f"\n  Guardado en: {CSV_SALIDA}")
    print(f"\n  SIGUIENTE PASO:")
    print(f"  Ejecuta extraer_features_gee.py sobre este nuevo CSV para")
    print(f"  añadir las variables satelitales antes de entrenar.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
