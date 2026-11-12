"""
diagnostico_modelo.py — Analiza la calidad del CSV de entrenamiento.
Ejecutar desde backend/:
    python diagnostico_modelo.py
"""
import pandas as pd
import numpy as np

CSV = "data/pozos_igme_final.csv"

df = pd.read_csv(CSV)

print("=" * 60)
print("  DIAGNÓSTICO DEL CSV DE ENTRENAMIENTO")
print("=" * 60)

print(f"\n📊 Total pozos: {len(df)}")

# Distribución de tiene_agua
n1 = df["tiene_agua"].sum()
n0 = (df["tiene_agua"] == 0).sum()
print(f"\n💧 Distribución tiene_agua:")
print(f"   Con agua (1): {n1} ({round(n1/len(df)*100)}%)")
print(f"   Secos    (0): {n0} ({round(n0/len(df)*100)}%)")

if n0 < 50:
    print("   ⚠️  MUY POCOS ejemplos negativos. El modelo no puede aprender a distinguir.")

# Geología
geo = df["geologia_score"]
print(f"\n🪨 geologia_score:")
print(f"   Valor 0.5 (sin dato real): {(geo == 0.5).sum()} pozos ({round((geo==0.5).sum()/len(df)*100)}%)")
print(f"   Valor 1.0 (favorable):     {(geo == 1.0).sum()} pozos")
print(f"   Valor 0.0 (desfavorable):  {(geo == 0.0).sum()} pozos")

if (geo == 0.5).sum() > len(df) * 0.8:
    print("   ⚠️  Más del 80% sin geología real. Variable poco útil para el modelo.")

# NDVI
ndvi = df["ndvi_verano"].dropna()
print(f"\n🌿 ndvi_verano:")
print(f"   Media:  {ndvi.mean():.3f}")
print(f"   Mín:    {ndvi.min():.3f}")
print(f"   Máx:    {ndvi.max():.3f}")
print(f"   Nulos:  {df['ndvi_verano'].isna().sum()}")

# Humedad
hum = df["humedad"].dropna()
print(f"\n💦 humedad ERA5:")
print(f"   Media:  {hum.mean():.4f}")
print(f"   Mín:    {hum.min():.4f}")
print(f"   Máx:    {hum.max():.4f}")
print(f"   Nulos:  {df['humedad'].isna().sum()}")

# Correlación con tiene_agua
print(f"\n📈 Correlación de cada variable con tiene_agua:")
features = ["ndvi_verano","ndvi_estabilidad","humedad","elevacion",
            "curvatura","acumulacion_flujo","geologia_score","lst_anomalia"]
for f in features:
    if f in df.columns:
        corr = df[f].corr(df["tiene_agua"])
        if pd.notna(corr):
            nivel = "🔴 muy baja" if abs(corr) < 0.05 else "🟡 baja" if abs(corr) < 0.15 else "🟢 útil"
            print(f"   {f:<25} {corr:+.3f}  {nivel}")

print(f"\n{'='*60}")
print("  DIAGNÓSTICO:")
print(f"{'='*60}")

problemas = []
if n0 < 50:
    problemas.append("❌ Muy pocos pozos secos. Necesitas más ejemplos negativos.")
if (geo == 0.5).sum() > len(df) * 0.8:
    problemas.append("❌ Geología sin datos reales para la mayoría de pozos.")
if df["ndvi_verano"].isna().sum() > 50:
    problemas.append("❌ Muchos pozos sin NDVI.")

if problemas:
    for p in problemas: print(f"  {p}")
else:
    print("  ✅ Los datos parecen correctos. El modelo puede mejorar con más pozos.")

print("=" * 60)
EOF