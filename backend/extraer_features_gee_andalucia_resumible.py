"""
extraer_features_gee_andalucia_resumible.py
Igual que extraer_features_gee_andalucia.py, pero con capacidad de
REANUDACIÓN: si existe un backup parcial de una ejecución anterior,
continúa exactamente desde donde se quedó en lugar de empezar de cero.

Ejecutar desde la carpeta backend/:
    python extraer_features_gee_andalucia_resumible.py

Para parar en cualquier momento: Ctrl+C en la terminal.
Para continuar otro día: vuelve a ejecutar el mismo comando.
"""

import ee
import json
import pandas as pd
import numpy as np
import time
from pathlib import Path

CSV_ENTRADA    = "data/pozos_andalucia_completo.csv"
CSV_SALIDA     = "data/pozos_andalucia_gee.csv"
CSV_BACKUP     = "data/pozos_andalucia_backup.csv"
GEE_KEY_PATH   = "gee_key.json"
BATCH_SIZE     = 20
PAUSA          = 2.0
MAX_REINTENTOS = 3
GUARDAR_CADA   = 25  # lotes


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
    if "ndvi_std" in df.columns and "ndvi_verano" in df.columns:
        media = df["ndvi_verano"].replace(0, np.nan)
        cv    = (df["ndvi_std"] / media).fillna(0)
        cv_n  = (cv - cv.min()) / (cv.max() - cv.min() + 1e-9)
        df["ndvi_estabilidad"] = (1 - cv_n).round(3)
    else:
        df["ndvi_estabilidad"] = 0.5

    if "curvatura_raw" in df.columns:
        c = df["curvatura_raw"]
        df["curvatura"] = ((c - c.min()) / (c.max() - c.min() + 1e-9)).round(3)
    else:
        df["curvatura"] = 0.0

    df["acumulacion_flujo"] = df["curvatura"]

    if "lst" in df.columns:
        lst     = df["lst"].fillna(df["lst"].median())
        lst_inv = lst.max() - lst
        df["lst_anomalia"] = ((lst_inv - lst_inv.min()) / (lst_inv.max() - lst_inv.min() + 1e-9)).round(3)
    else:
        df["lst_anomalia"] = 0.0

    if "elevacion_gee" in df.columns:
        df["elevacion"] = df["elevacion_gee"].fillna(df["elevacion"])

    df.drop(columns=["ndvi_std", "elevacion_gee", "curvatura_raw", "lst"], errors="ignore", inplace=True)
    return df


def cargar_estado_inicial():
    """
    Comprueba si existe un backup parcial de una ejecución anterior.
    Si existe, lo usa como punto de partida y calcula desde qué pozo
    continuar. Si no existe, carga el CSV original desde cero.
    """
    cols_gee = ["ndvi_verano", "ndvi_std", "humedad", "elevacion_gee", "curvatura_raw", "lst"]

    if Path(CSV_BACKUP).exists():
        print(f"[REANUDAR] Se encontró un backup parcial: {CSV_BACKUP}")
        df = pd.read_csv(CSV_BACKUP)

        # Asegurar que existen todas las columnas necesarias
        for col in cols_gee:
            if col not in df.columns:
                df[col] = np.nan

        # Calcular cuántos pozos ya tienen datos GEE completos (procesados con éxito)
        ya_procesados = df["ndvi_verano"].notna().sum()

        # El punto de reanudación es el primer índice SIN dato de ndvi_verano,
        # pero retrocedemos al inicio del lote de BATCH_SIZE para no
        # fragmentar lotes a medias.
        primer_pendiente = df[df["ndvi_verano"].isna()].index.min()
        if pd.isna(primer_pendiente):
            print("[REANUDAR] El backup ya tiene TODOS los pozos procesados.")
            inicio = len(df)
        else:
            inicio = (primer_pendiente // BATCH_SIZE) * BATCH_SIZE

        print(f"[REANUDAR] {ya_procesados}/{len(df)} pozos ya completados.")
        print(f"[REANUDAR] Continuando desde el pozo {inicio + 1}...\n")
        return df, inicio

    else:
        print(f"[NUEVO] No hay backup previo. Empezando desde cero con {CSV_ENTRADA}")
        df = pd.read_csv(CSV_ENTRADA)
        for col in cols_gee:
            df[col] = np.nan
        return df, 0


def main():
    print("=" * 60)
    print("  EXTRACCIÓN GEE — POZOS ANDALUCÍA (REANUDABLE)")
    print("=" * 60)

    init_gee()

    df, inicio = cargar_estado_inicial()
    total = len(df)

    if inicio >= total:
        print("\n✅ No queda nada pendiente. El proceso ya estaba completo.")
        df_final = postprocesar(df.copy())
        Path("data").mkdir(exist_ok=True)
        df_final.to_csv(CSV_SALIDA, index=False)
        print(f"Guardado final en: {CSV_SALIDA}")
        return

    print("[GEE] Construyendo imágenes satelitales...")
    combinada = construir_imagenes()
    print("[GEE] Listo. Iniciando extracción...\n")

    cols_gee  = ["ndvi_verano", "ndvi_std", "humedad", "elevacion_gee", "curvatura_raw", "lst"]
    n_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    batch_inicial = inicio // BATCH_SIZE
    errores = 0

    try:
        for b in range(batch_inicial, n_batches):
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

            if resultados:
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

            if (b + 1) % GUARDAR_CADA == 0:
                df.to_csv(CSV_BACKUP, index=False)
                print(f"    💾 Backup parcial guardado ({end}/{total} pozos procesados)")

            time.sleep(PAUSA)

    except KeyboardInterrupt:
        print("\n\n⏸️  Proceso interrumpido manualmente (Ctrl+C).")
        df.to_csv(CSV_BACKUP, index=False)
        print(f"💾 Backup guardado antes de salir. Puedes continuar mañana")
        print(f"   ejecutando este mismo script — reanudará automáticamente.")
        return

    # Guardar backup final siempre, por si el proceso se corta después
    df.to_csv(CSV_BACKUP, index=False)

    print("\n[POST] Calculando estabilidad NDVI y normalizando...")
    df_final = postprocesar(df.copy())

    Path("data").mkdir(exist_ok=True)
    df_final.to_csv(CSV_SALIDA, index=False)

    n_ok = df["ndvi_verano"].notna().sum()
    print(f"\n{'='*60}")
    print(f"  ✅ COMPLETADO")
    print(f"  Total pozos:       {total}")
    print(f"  Con datos GEE:     {n_ok}")
    print(f"  Batches con error: {errores}")
    print(f"  Fichero guardado:  {CSV_SALIDA}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
