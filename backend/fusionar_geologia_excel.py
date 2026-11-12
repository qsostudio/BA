"""
fusionar_geologia_excel.py
Lee el Excel original del IGME, extrae la litología real de cada pozo
y la fusiona con los datos GEE que ya tienes en pozos_igme_final.csv.

Ejecutar desde backend/:
    python fusionar_geologia_excel.py ruta_al_excel.xlsx

Ejemplo:
    python fusionar_geologia_excel.py "C:/Users/enriq/Downloads/BD_Puntos_Agua_v2_0.xlsx"
"""

import sys
import pandas as pd
import numpy as np

ROCAS_BUENAS = [
    "caliza","calcarenita","dolomia","dolomita","mármol","travertino",
    "arenisca","conglomerado","grava","arena","aluvial","cuaternario",
    "terciario","detrítico","carbonato","arcosa","fango","limo","brecha",
    "calcoesquisto","pizarra calcárea","arenoso","areniscas",
]
ROCAS_MALAS = [
    "granito","granodiorita","diorita","gabro","basalto","riolita",
    "andesita","pórfido","esquisto","gneis","filita","pizarra","cuarcita",
    "cuarzo","migmatita","anfibolita","serpentinita","peridotita",
    "metamórfico","plutónico","volcánico","intrusivo",
]

def score_lith(lith):
    if not lith or str(lith).strip() in ["","nan","None","-"]:
        return 0.5
    lit = str(lith).lower()
    if any(r in lit for r in ROCAS_BUENAS):
        return 1.0
    elif any(r in lit for r in ROCAS_MALAS):
        return 0.0
    return 0.5

def main():
    if len(sys.argv) < 2:
        print("Uso: python fusionar_geologia_excel.py ruta_excel.xlsx")
        print("Ejemplo:")
        print('  python fusionar_geologia_excel.py "C:/Users/enriq/Downloads/BD_Puntos_Agua_v2_0.xlsx"')
        return

    excel_path = sys.argv[1]
    csv_gee    = "data/pozos_igme_final.csv"
    csv_salida = "data/pozos_igme_con_geologia.csv"

    print("=" * 60)
    print("  FUSIONAR GEOLOGÍA EXCEL + DATOS GEE")
    print("=" * 60)

    # Leer Excel original
    print(f"\n[1] Leyendo Excel: {excel_path}")
    df_excel = pd.read_excel(excel_path, skiprows=1)
    print(f"    Columnas: {list(df_excel.columns)}")
    print(f"    Filas: {len(df_excel)}")

    # Buscar columna de litología
    lith_col = None
    for c in df_excel.columns:
        if any(k in str(c).lower() for k in ["lito","litol","roca","geol","litología","material"]):
            lith_col = c
            print(f"    ✅ Columna litología: '{lith_col}'")
            break

    if not lith_col:
        print("    ⚠️  No se encontró columna de litología explícita.")
        print("    Columnas disponibles:", list(df_excel.columns))
        # Intentar con todas las columnas de texto
        for c in df_excel.columns:
            if df_excel[c].dtype == object and c not in ["Provincia","Municipio","Demarcación","Naturaleza","Utilización"]:
                print(f"    Posible candidata: '{c}' — muestra: {df_excel[c].dropna().head(3).tolist()}")

    # Coordenadas del Excel
    lat_col = next((c for c in df_excel.columns if "coord" in str(c).lower() and "y" in str(c).lower() and "etrs" in str(c).lower()), None)
    lon_col = next((c for c in df_excel.columns if "coord" in str(c).lower() and "x" in str(c).lower() and "etrs" in str(c).lower()), None)

    if not lat_col or not lon_col:
        # Buscar por nombre aproximado
        for c in df_excel.columns:
            cl = str(c).lower()
            if ("y" in cl or "lat" in cl) and ("etrs" in cl or "coord" in cl):
                lat_col = c
            if ("x" in cl or "lon" in cl) and ("etrs" in cl or "coord" in cl):
                lon_col = c

    print(f"    Columna lat: '{lat_col}' | Columna lon: '{lon_col}'")

    # Leer CSV con datos GEE
    print(f"\n[2] Leyendo datos GEE: {csv_gee}")
    df_gee = pd.read_csv(csv_gee)
    print(f"    {len(df_gee)} pozos con datos satelitales")

    # Si tenemos litología, calcular score y fusionar por coordenadas
    if lith_col and lat_col and lon_col:
        df_excel["lat_r"] = pd.to_numeric(df_excel[lat_col], errors="coerce").round(4)
        df_excel["lon_r"] = pd.to_numeric(df_excel[lon_col], errors="coerce").round(4)
        df_excel["geo_score_igme"] = df_excel[lith_col].apply(score_lith)

        df_gee["lat_r"] = df_gee["lat"].round(4)
        df_gee["lon_r"] = df_gee["lon"].round(4)

        df_merged = df_gee.merge(
            df_excel[["lat_r","lon_r","geo_score_igme",lith_col]],
            on=["lat_r","lon_r"], how="left"
        )

        n_matched = df_merged["geo_score_igme"].notna().sum()
        print(f"\n[3] Fusión: {n_matched}/{len(df_gee)} pozos emparejados con litología real")

        # Actualizar geologia_score donde hay dato real
        df_merged.loc[df_merged["geo_score_igme"].notna(), "geologia_score"] = df_merged["geo_score_igme"]
        df_merged.drop(columns=["lat_r","lon_r","geo_score_igme"], errors="ignore", inplace=True)

        n1 = (df_merged["geologia_score"] == 1.0).sum()
        n0 = (df_merged["geologia_score"] == 0.0).sum()
        n5 = (df_merged["geologia_score"] == 0.5).sum()
        print(f"\n  Favorable (1.0): {n1} pozos ({round(n1/len(df_merged)*100)}%)")
        print(f"  Desfavorable (0.0): {n0} pozos ({round(n0/len(df_merged)*100)}%)")
        print(f"  Sin dato (0.5): {n5} pozos ({round(n5/len(df_merged)*100)}%)")

        df_merged.to_csv(csv_salida, index=False)

    else:
        print("\n⚠️  No se pudo fusionar. Guardando CSV sin cambios.")
        df_gee.to_csv(csv_salida, index=False)

    print(f"\n✅ Guardado: {csv_salida}")
    print("   Sube este fichero desde el panel de entrenamiento.")
    print("=" * 60)

if __name__ == "__main__":
    main()
