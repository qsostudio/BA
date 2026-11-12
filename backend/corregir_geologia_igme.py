"""
corregir_geologia_igme.py
Usa la columna 'litologia' del propio IGME para calcular
geologia_score en lugar de Macrostrat, que no tiene cobertura
en muchas zonas de España.

Ejecutar desde backend/:
    python corregir_geologia_igme.py
"""

import pandas as pd

CSV_ENTRADA = "data/pozos_igme_final.csv"
CSV_SALIDA  = "data/pozos_igme_corregido.csv"

# Clasificación de litologías del IGME en español
ROCAS_BUENAS = [
    "caliza", "calcarenita", "dolomia", "dolomita", "mármol",
    "arenisca", "conglomerado", "grava", "arena", "aluvial",
    "cuaternario", "terciario", "detrítico", "carbonato",
    "travertino", "brecha", "arcosa", "fango", "limo",
    "pizarra calcárea", "calcoesquisto",
]

ROCAS_MALAS = [
    "granito", "granodiorita", "diorita", "gabro", "basalto",
    "riolita", "andesita", "pórfido", "esquisto", "gneis",
    "filita", "pizarra", "cuarcita", "cuarzo", "migmatita",
    "anfibolita", "serpentinita", "peridotita",
]


def clasificar_igme(litologia: str) -> float:
    if not litologia or str(litologia).strip() in ["", "nan", "None"]:
        return 0.5
    lit = str(litologia).lower()
    if any(r in lit for r in ROCAS_BUENAS):
        return 1.0
    elif any(r in lit for r in ROCAS_MALAS):
        return 0.0
    return 0.5


def main():
    print("=" * 60)
    print("  CORREGIR GEOLOGÍA CON DATOS DEL IGME")
    print("=" * 60)

    df = pd.read_csv(CSV_ENTRADA)
    print(f"\n[DATOS] {len(df)} pozos cargados")

    # Ver qué columna de litología existe
    lith_col = None
    for c in df.columns:
        if "lito" in c.lower() or "roca" in c.lower() or "litol" in c.lower():
            lith_col = c
            break

    if lith_col:
        print(f"[INFO] Usando columna: '{lith_col}'")
        print(f"\n[INFO] Litologías más frecuentes:")
        top = df[lith_col].value_counts().head(10)
        for lith, count in top.items():
            score = clasificar_igme(str(lith))
            emoji = "✅" if score == 1.0 else "❌" if score == 0.0 else "⚪"
            print(f"  {emoji} {lith}: {count} pozos → score {score}")

        df["geologia_score"] = df[lith_col].apply(clasificar_igme)
    else:
        print("[AVISO] No se encontró columna de litología en el CSV.")
        print("        Columnas disponibles:", list(df.columns))
        print("        Se mantendrá geologia_score como está.")

    # Estadísticas finales
    n_buena = (df["geologia_score"] == 1.0).sum()
    n_mala  = (df["geologia_score"] == 0.0).sum()
    n_media = (df["geologia_score"] == 0.5).sum()

    print(f"\n[RESULTADO] Distribución de geología:")
    print(f"  ✅ Favorable (1.0): {n_buena} pozos ({round(n_buena/len(df)*100)}%)")
    print(f"  ❌ Desfavorable (0.0): {n_mala} pozos ({round(n_mala/len(df)*100)}%)")
    print(f"  ⚪ Media/sin dato (0.5): {n_media} pozos ({round(n_media/len(df)*100)}%)")

    df.to_csv(CSV_SALIDA, index=False)
    print(f"\n[OK] Guardado: {CSV_SALIDA}")
    print("\n  SIGUIENTE PASO:")
    print(f"  Sube {CSV_SALIDA} desde el panel de entrenamiento.")
    print("=" * 60)


if __name__ == "__main__":
    main()
