"""
extraer_features_gee.py
Extrae NDVI, humedad, elevación, curvatura y temperatura desde GEE
para cada pozo del CSV del IGME. Batch de 20 puntos con 3 reintentos.

Ejecutar desde la carpeta backend/:
    python extraer_features_gee.py
"""

import ee
import json
import pandas as pd
import numpy as np
import time
from pathlib import Path

CSV_ENTRADA    = "data/pozos_igme_convertidos.csv"
CSV_SALIDA     = "data/pozos_igme_completos.csv"
CSV_BACKUP     = "data/pozos_igme_backup.csv"
GEE_KEY_PATH   = "gee_key.json"
BATCH_SIZE     = 20
PAUSA          = 2.0
MAX_REINTENTOS = 3


def init_gee():
    with open(GEE_KEY_PATH) as f:
        key_data = json.load(f)
    from google.oauth2 import service_account as google_sa
    scopes = ["https://www.googleapis.com/auth/earthengine"]
    creds  = google_sa.Credentials.from_service_account_info(key_data, scopes=scopes)
    ee.Initialize(creds)
    print("[GEE] Inicializado correctamente")


def construir_imagenes():
    ndvi_col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate("2021-01-01", "2024-12-31")
        .filter(ee.Filter.calendarRange(6, 8, "month"))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        .map(lambda img: img.normalizedDifference(["B8", "B4"]).rename("NDVI"))
    )
    ndvi_mean = ndvi_col.mean().rename("ndvi_verano")
    ndvi_std  = ndvi_col.reduce(ee.Reducer.stdDev()).rename("ndvi_std")

    humedad = (
        ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
        .filterDate("2021-01-01", "2024-12-31")
        .filter(ee.Filter.calendarRange(6, 8, "month"))
        .select("volumetric_soil_water_layer_1")
        .mean()
        .rename("humedad")
    )

    dem       = ee.Image("USGS/SRTMGL1_003").select("elevation").rename("elevacion_gee")
    curvatura = dem.convolve(ee.Kernel.laplacian8()).multiply(-1).rename("curvatura_raw")

    def a_celsius(img):
        return img.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15).rename("LST")

    lst = (
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        .merge(ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"))
        .filterDate("2021-06-01", "2024-08-31")
        .filter(ee.Filter.calendarRange(6, 8, "month"))
        .filter(ee.Filter.lt("CLOUD_COVER", 20))
        .map(a_celsius)
        .mean()
        .rename("lst")
    )

    return (ndvi_mean
            .addBands(ndvi_std)
            .addBands(humedad)
            .addBands(dem)
            .addBands(curvatura)
            .addBands(lst))


def extraer_batch(combinada, lats, lons, indices):
    features = [
        ee.Feature(ee.Geometry.Point([float(lon), float(lat)]), {"idx": int(idx)})
        for idx, lat, lon in zip(indices, lats, lons)
        if not (np.isnan(float(lat)) or np.isnan(float(lon)))
    ]
    if not features:
        return []

    fc = ee.FeatureCollection(features)

    def extraer(feat):
        vals = combinada.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=feat.geometry(),
            scale=30,
            maxPixels=1e6
        )
        return feat.set(vals)

    return fc.map(extraer).getInfo()["features"]


def postprocesar(df):
    # ndvi_estabilidad: inverso del coeficiente de variación
    if "ndvi_std" in df.columns and "ndvi_verano" in df.columns:
        media = df["ndvi_verano"].replace(0, np.nan)
        cv    = (df["ndvi_std"] / media).fillna(0)
        cv_n  = (cv - cv.min()) / (cv.max() - cv.min() + 1e-9)
        df["ndvi_estabilidad"] = (1 - cv_n).round(3)
    else:
        df["ndvi_estabilidad"] = 0.5

    # curvatura normalizada
    if "curvatura_raw" in df.columns:
        c = df["curvatura_raw"]
        df["curvatura"] = ((c - c.min()) / (c.max() - c.min() + 1e-9)).round(3)
    else:
        df["curvatura"] = 0.0

    df["acumulacion_flujo"] = df["curvatura"]

    # lst anomalía (zonas más frescas = mayor anomalía = posible acuífero)
    if "lst" in df.columns:
        lst     = df["lst"].fillna(df["lst"].median())
        lst_inv = lst.max() - lst
        df["lst_anomalia"] = ((lst_inv - lst_inv.min()) / (lst_inv.max() - lst_inv.min() + 1e-9)).round(3)
    else:
        df["lst_anomalia"] = 0.0

    df["geologia_score"] = 0.5

    # usar elevacion del IGME si la de GEE falla
    if "elevacion_gee" in df.columns:
        df["elevacion"] = df["elevacion_gee"].fillna(df["elevacion"])

    # borrar columnas intermedias
    df.drop(columns=["ndvi_std", "elevacion_gee", "curvatura_raw", "lst"], errors="ignore", inplace=True)
    return df


def main():
    print("=" * 60)
    print("  EXTRACCIÓN GEE — POZOS IGME")
    print("=" * 60)

    init_gee()

    if not Path(CSV_ENTRADA).exists():
        print(f"\n❌ No se encuentra {CSV_ENTRADA}")
        return

    df = pd.read_csv(CSV_ENTRADA)
    print(f"\n[DATOS] {len(df)} pozos cargados")

    print("[GEE] Construyendo imágenes satelitales...")
    combinada = construir_imagenes()
    print("[GEE] Listo. Iniciando extracción...\n")

    for col in ["ndvi_verano", "ndvi_std", "humedad", "elevacion_gee", "curvatura_raw", "lst"]:
        df[col] = np.nan

    total     = len(df)
    n_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    errores   = 0

    for b in range(n_batches):
        start    = b * BATCH_SIZE
        end      = min(start + BATCH_SIZE, total)
        batch_df = df.iloc[start:end]
        lats     = batch_df["lat"].values
        lons     = batch_df["lon"].values
        indices  = batch_df.index.values

        print(f"  Batch {b+1}/{n_batches} (pozos {start+1}-{end})...", end=" ", flush=True)

        resultados = None
        for intento in range(MAX_REINTENTOS):
            try:
                resultados = extraer_batch(combinada, lats, lons, indices)
                break
            except Exception as e:
                if intento < MAX_REINTENTOS - 1:
                    espera = 5 * (intento + 1)
                    print(f"\n    ⏳ Reintento {intento+1}/{MAX_REINTENTOS-1} (espera {espera}s)...", end=" ", flush=True)
                    time.sleep(espera)
                else:
                    errores += 1
                    print(f"⚠️  Fallo tras {MAX_REINTENTOS} intentos: {e}")
                    df.to_csv(CSV_BACKUP, index=False)

        if resultados:
            cols_gee = ["ndvi_verano", "ndvi_std", "humedad", "elevacion_gee", "curvatura_raw", "lst"]
            for feat in resultados:
                props = feat["properties"]
                idx   = int(props.get("idx", -1))
                if idx < 0:
                    continue
                for col in cols_gee:
                    val = props.get(col)
                    if val is not None:
                        df.at[idx, col] = round(float(val), 4)
            print(f"✅ ({len(resultados)} pts)")

        time.sleep(PAUSA)

    print("\n[POST] Calculando estabilidad NDVI y normalizando...")
    df = postprocesar(df)

    Path("data").mkdir(exist_ok=True)
    df.to_csv(CSV_SALIDA, index=False)

    n_ok = df["ndvi_verano"].notna().sum()
    print(f"\n{'='*60}")
    print(f"  ✅ COMPLETADO")
    print(f"  Total pozos:       {total}")
    print(f"  Con datos GEE:     {n_ok}")
    print(f"  Batches con error: {errores}")
    print(f"  Fichero guardado:  {CSV_SALIDA}")
    print(f"\n  SIGUIENTE PASO:")
    print(f"  Sube {CSV_SALIDA} desde el panel de entrenamiento de la app.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
