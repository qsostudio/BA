"""
mejorar_dataset.py
Añade profundidad normalizada al CSV de entrenamiento y
elimina geologia_score (que está vacío) del conjunto de features.

Ejecutar desde backend/:
    python mejorar_dataset.py "ruta/al/BD_Puntos_Agua_v2_0.xlsx"
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

CSV_GEE    = "data/pozos_igme_final.csv"
CSV_SALIDA = "data/pozos_entrenamiento_final.csv"


def main():
    if len(sys.argv) < 2:
        print("Uso: python mejorar_dataset.py ruta_excel.xlsx")
        return

    excel_path = sys.argv[1]

    print("=" * 60)
    print("  MEJORAR DATASET DE ENTRENAMIENTO")
    print("=" * 60)

    # Leer Excel original
    df_excel = pd.read_excel(excel_path, skiprows=1)
    print(f"\n[1] Excel: {len(df_excel)} pozos")

    # Extraer coordenadas y profundidad del Excel
    df_excel["lat"] = pd.to_numeric(df_excel["Coordenada Y (ETRS89)"], errors="coerce").round(4)
    df_excel["lon"] = pd.to_numeric(df_excel["Coordenada X (ETRS89)"], errors="coerce").round(4)
    df_excel["profundidad_excel"] = pd.to_numeric(df_excel["Profundidad(m)"], errors="coerce")

    # Codificar naturaleza
    def cod_naturaleza(n):
        if "sondeo" in str(n).lower(): return 0.7
        if "galería" in str(n).lower(): return 0.8
        return 1.0  # Pozo estándar
    df_excel["tipo_captacion"] = df_excel["Naturaleza"].apply(cod_naturaleza)

    df_excel = df_excel[["lat","lon","profundidad_excel","tipo_captacion"]].dropna(subset=["lat","lon"])

    # Leer CSV con datos GEE
    df_gee = pd.read_csv(CSV_GEE)
    print(f"[2] CSV GEE: {len(df_gee)} pozos")

    df_gee["lat_r"] = df_gee["lat"].round(4)
    df_gee["lon_r"] = df_gee["lon"].round(4)
    df_excel["lat_r"] = df_excel["lat"].round(4)
    df_excel["lon_r"] = df_excel["lon"].round(4)

    df_merged = df_gee.merge(
        df_excel[["lat_r","lon_r","profundidad_excel","tipo_captacion"]],
        on=["lat_r","lon_r"], how="left"
    )
    df_merged.drop(columns=["lat_r","lon_r"], inplace=True)

    n_prof = df_merged["profundidad_excel"].notna().sum()
    print(f"[3] Profundidad emparejada: {n_prof}/{len(df_merged)} pozos")

    # Usar profundidad del Excel si está disponible, si no la del CSV original
    df_merged["profundidad"] = df_merged["profundidad_excel"].fillna(df_merged["profundidad"])
    df_merged.drop(columns=["profundidad_excel"], inplace=True)

    # Normalizar profundidad 0-1 (más profundo = más difícil = menor prob. agua superficial)
    p_max = df_merged["profundidad"].quantile(0.95)
    p_min = df_merged["profundidad"].min()
    df_merged["profundidad_norm"] = (
        (df_merged["profundidad"].fillna(df_merged["profundidad"].median()) - p_min)
        / (p_max - p_min + 1e-9)
    ).clip(0, 1).round(3)

    # tipo_captacion: rellenar con 1.0 si no se emparejó
    df_merged["tipo_captacion"] = df_merged["tipo_captacion"].fillna(1.0)

    # Estadísticas finales
    print(f"\n[STATS] profundidad_norm:")
    print(f"  Media: {df_merged['profundidad_norm'].mean():.3f}")
    print(f"  Mín:   {df_merged['profundidad_norm'].min():.3f}")
    print(f"  Máx:   {df_merged['profundidad_norm'].max():.3f}")

    print(f"\n[STATS] tiene_agua:")
    n1 = df_merged["tiene_agua"].sum()
    n0 = (df_merged["tiene_agua"]==0).sum()
    print(f"  Con agua: {n1} ({round(n1/len(df_merged)*100)}%)")
    print(f"  Secos:    {n0} ({round(n0/len(df_merged)*100)}%)")

    # Guardar
    df_merged.to_csv(CSV_SALIDA, index=False)
    print(f"\n✅ Guardado: {CSV_SALIDA}")
    print("\n  SIGUIENTE PASO:")
    print("  1. Copia modelo_ia_v2.py a backend/services/modelo_ia.py")
    print(f"  2. Sube {CSV_SALIDA} desde el panel de entrenamiento")
    print("=" * 60)


if __name__ == "__main__":
    main()
