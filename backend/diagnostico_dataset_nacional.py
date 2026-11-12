"""
diagnostico_dataset_nacional.py
Analiza el CSV descargado del IGME para decidir cuántos pozos merece
la pena procesar con Google Earth Engine antes de lanzar el proceso
largo de extracción de variables satelitales.

Ejecutar desde backend/:
    python diagnostico_dataset_nacional.py
"""

import pandas as pd

CSV = "data/pozos_igme_nacional.csv"

df = pd.read_csv(CSV)

print("=" * 60)
print("  DIAGNÓSTICO DEL DATASET NACIONAL")
print("=" * 60)

print(f"\n📊 Total de filas: {len(df)}")
print(f"📋 Columnas: {list(df.columns)}")

# Coordenadas válidas
validos = df.dropna(subset=["lat", "lon"])
print(f"\n📍 Con coordenadas válidas: {len(validos)} ({round(len(validos)/len(df)*100)}%)")

# Naturaleza del punto (pozo, sondeo, manantial, galería...)
if "naturaleza" in df.columns:
    print(f"\n🔍 Distribución por tipo de captación (Naturaleza):")
    print(df["naturaleza"].value_counts().head(15).to_string())

# Utilización (si está en uso o no)
if "utilizacion" in df.columns:
    print(f"\n💧 Distribución por utilización:")
    print(df["utilizacion"].value_counts(dropna=False).head(15).to_string())
    print(f"\n   Nulos en utilización: {df['utilizacion'].isna().sum()} ({round(df['utilizacion'].isna().sum()/len(df)*100)}%)")

# Profundidad
if "profundidad" in df.columns:
    prof = pd.to_numeric(df["profundidad"], errors="coerce")
    print(f"\n📏 Profundidad:")
    print(f"   Con dato: {prof.notna().sum()} ({round(prof.notna().sum()/len(df)*100)}%)")
    print(f"   Media: {prof.mean():.1f} m | Mediana: {prof.median():.1f} m | Máx: {prof.max():.1f} m")

# Duplicados por coordenadas (mismo punto registrado varias veces)
dup = validos.duplicated(subset=["lat", "lon"]).sum()
print(f"\n🔁 Coordenadas duplicadas: {dup} ({round(dup/len(validos)*100)}%)")

# Provincias
if "provincia" in df.columns:
    print(f"\n🗺️  Pozos por provincia:")
    print(df["provincia"].value_counts().to_string())

print(f"\n{'='*60}")
print("  RECOMENDACIÓN")
print(f"{'='*60}")

# Sugerir un subconjunto razonable: solo los que tienen
# naturaleza relevante (pozo/sondeo) y utilización conocida
if "naturaleza" in df.columns and "utilizacion" in df.columns:
    filtro = (
        validos["naturaleza"].isin(["Pozo", "Sondeo", "Pozo excavado", "Sondeo de investigación"])
        & validos["utilizacion"].notna()
    )
    candidatos = validos[filtro].drop_duplicates(subset=["lat", "lon"])
    print(f"\n  Pozos/sondeos con utilización conocida y sin duplicar: {len(candidatos)}")
    print(f"  Este sería un conjunto de calidad razonable para extraer variables GEE.")
    print(f"\n  Tiempo estimado de extracción GEE (a 20 pozos/lote, ~10s/lote):")
    print(f"  {round(len(candidatos)/20 * 10 / 60)} minutos aproximadamente")
else:
    print("\n  No se pudieron aplicar los filtros automáticos. Revisa las columnas disponibles arriba.")

print("=" * 60)
