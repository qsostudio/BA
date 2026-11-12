"""
pdf_service.py  -  Genera el informe PDF del análisis de una parcela.

Para cada resultado (punto óptimo, vaguadas, escorrentías, caudal,
geología y lo que calculan los dos modelos de inteligencia artificial)
explica DOS cosas:
  1. Qué fuente de datos se ha consultado.
  2. Qué cálculo matemático se ha aplicado.

Usa fpdf2 (pip install fpdf2)  -  librería pura Python, sin dependencias
de sistema (a diferencia de weasyprint/reportlab con Cairo/Pango), para
minimizar problemas de instalación en Windows.
"""

from datetime import datetime
from io import BytesIO
from fpdf import FPDF
from fpdf.fonts import FontFace

AZUL      = (30, 64, 175)
AZUL_CL   = (219, 234, 254)
GRIS      = (100, 100, 100)
NEGRO     = (20, 20, 20)
VERDE     = (21, 128, 61)
NARANJA   = (194, 120, 3)
ESTRELLA_NEGRA  = (17, 24, 39)     # mismo color que usa el mapa para BALSA
ESTRELLA_DORADA = (212, 175, 55)   # mismo color que usa el mapa para POZO


def _fmt(v, decimales=2, sufijo=""):
    """Formatea un número con seguridad, o ' - ' si es None."""
    if v is None:
        return " - "
    try:
        return f"{float(v):.{decimales}f}{sufijo}"
    except (TypeError, ValueError):
        return str(v)


def _limpiar_texto(texto) -> str:
    """
    Sustituye caracteres Unicode fuera de Latin-1 por equivalentes seguros.
    Necesario porque datos que vienen de fuentes externas (nombres
    geológicos del IGME/Macrostrat, etiquetas de ign_service.py) pueden
    traer rayas, comillas tipográficas u otros símbolos que la fuente
    básica de fpdf2 no soporta y harían fallar la generación del PDF.
    """
    if not isinstance(texto, str):
        texto = str(texto)
    reemplazos = {
        "—": " - ", "–": "-", "…": "...", "•": "-",
        "⚠": "[!]", "→": "->", "≥": ">=", "≤": "<=",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    }
    for viejo, nuevo in reemplazos.items():
        texto = texto.replace(viejo, nuevo)
    # Red de seguridad final: cualquier carácter que siga sin caber en
    # Latin-1 (emojis, símbolos raros) se elimina en vez de romper el PDF.
    return texto.encode("latin-1", "ignore").decode("latin-1")


class InformePDF(FPDF):
    def __init__(self):
        super().__init__(format="A4", unit="mm")
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(16, 16, 16)

    # Todo texto pasa por _limpiar_texto() automáticamente, venga de donde
    # venga — así no hay que acordarse de limpiar cada valor a mano.
    def cell(self, w=None, h=None, text="", *args, **kwargs):
        return super().cell(w, h, _limpiar_texto(text), *args, **kwargs)

    def multi_cell(self, w, h=None, text="", *args, **kwargs):
        return super().multi_cell(w, h, _limpiar_texto(text), *args, **kwargs)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GRIS)
        self.cell(140, 8, "Buscador de Agua Subterránea  -  Informe de análisis", align="L")
        self.cell(38, 8, datetime.now().strftime("%d/%m/%Y"), align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*GRIS)
        self.line(16, 18, 194, 18)
        self.ln(4)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*GRIS)
        self.cell(0, 8, f"Página {self.page_no()}", align="C")

    def titulo_seccion(self, num, texto):
        self.ln(3)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*AZUL)
        self.cell(0, 9, f"{num}. {texto}", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*AZUL)
        self.set_line_width(0.6)
        self.line(16, self.get_y(), 194, self.get_y())
        self.set_line_width(0.2)
        self.ln(4)
        self.set_text_color(*NEGRO)

    def subtitulo(self, texto):
        self.set_font("Helvetica", "B", 10.5)
        self.set_text_color(*NEGRO)
        self.cell(0, 7, texto, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def indicador_estrella(self, color_rgb, etiqueta):
        """Cuadro de color + etiqueta -- sustituye al carácter ★ (fuera del
        rango Latin-1 que soporta la fuente básica) para distinguir en el
        PDF el punto de BALSA (negro) del de POZO (dorado), igual que en
        el mapa de la app."""
        self.set_fill_color(*color_rgb)
        self.set_draw_color(60, 60, 60)
        self.cell(6, 6, "", fill=True, border=1)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*NEGRO)
        self.cell(0, 6, f"  {etiqueta}", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def parrafo(self, texto):
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*NEGRO)
        self.multi_cell(0, 5.2, texto)
        self.ln(1)

    def dato_consultado(self, fuente, calculo):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*AZUL)
        self.cell(0, 5.5, "Fuente:", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*NEGRO)
        self.multi_cell(0, 5.2, fuente)
        self.ln(1)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*AZUL)
        self.cell(0, 5.5, "Cálculo:", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*NEGRO)
        self.multi_cell(0, 5.2, calculo)
        self.ln(3)

    def aviso(self, texto):
        self.set_fill_color(*AZUL_CL)
        self.set_font("Helvetica", "I", 8.5)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 5, "  [!] " + texto, fill=True)
        self.ln(2)
        self.set_text_color(*NEGRO)


def _portada(pdf: InformePDF, coords, fecha_txt):
    pdf.add_page()
    pdf.ln(30)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(*AZUL)
    pdf.cell(0, 14, "Informe de Análisis Hidrogeológico", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(*GRIS)
    pdf.cell(0, 10, "Buscador de Agua Subterránea", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)

    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    lat_c, lon_c = sum(lats) / len(lats), sum(lons) / len(lons)

    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(*NEGRO)
    pdf.cell(0, 7, f"Fecha del informe: {fecha_txt}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Centro de la parcela: {lat_c:.6f}, {lon_c:.6f}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Vértices del polígono analizado: {len(coords)}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(14)

    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(*GRIS)
    pdf.multi_cell(
        0, 5,
        "Este informe explica, para cada resultado mostrado en la aplicación, qué fuente de datos "
        "se ha consultado y qué cálculo matemático se ha aplicado  -  de forma que cada número pueda "
        "auditarse y no se tome como una \"caja negra\".",
        align="C",
    )


def _num(v, decimales=1):
    """Número con coma decimal, para los textos del informe."""
    return f"{float(v):.{decimales}f}".replace(".", ",")


def _texto_profundidad(prof: dict) -> str:
    return (f"~{prof['profundidad_estimada_m']:.0f} m (entre {prof['intervalo_min_m']:.0f} y "
            f"{prof['intervalo_max_m']:.0f} m, {prof.get('confianza_pct', 80)} % de confianza)")


def _texto_agua(agua: dict) -> str:
    if agua.get("n_masas", 0) == 0:
        return "No se detecta agua en la parcela ni a menos de 150 m"
    return (f"{_num(agua.get('area_agua_parcela_ha', 0), 2)} ha dentro de la parcela; "
            f"{agua.get('n_masas', 0)} masa(s) de agua en la parcela y 150 m alrededor")


def _seccion_resumen(pdf: InformePDF, gee: dict, vaguadas: dict, geologia: dict, pozo: dict = None, ia: dict = None):
    pdf.add_page()
    pdf.titulo_seccion(1, "Resumen del análisis")
    pdf.parrafo(
        "Tabla con los resultados principales. Cada uno se explica en detalle, con su fuente y su "
        "fórmula, en las secciones siguientes."
    )

    ia = ia or {}
    filas = [
        ("Punto óptimo BALSA, estrella negra (lat, lon)", f"{gee.get('pozo_lat', 0):.6f}, {gee.get('pozo_lon', 0):.6f}"),
        ("Índice relativo BALSA (0-100; solo comparable dentro de esta parcela)", _fmt(gee.get("pozo_score"), 1, " / 100")),
    ]
    if pozo and pozo.get("disponible"):
        filas += [
            ("Punto óptimo POZO, estrella dorada (lat, lon)",
             f"{pozo.get('pozo_profundo_lat', 0):.6f}, {pozo.get('pozo_profundo_lon', 0):.6f}"),
            ("Índice relativo POZO (0-100; solo comparable dentro de esta parcela)",
             _fmt(pozo.get("pozo_profundo_score"), 1, " / 100")),
        ]
    prof = ia.get("profundidad") or {}
    if prof.get("disponible"):
        filas.append(("IA - Profundidad estimada del agua subterránea en el punto POZO", _texto_profundidad(prof)))
    agua = ia.get("agua") or {}
    if agua.get("disponible"):
        filas.append(("IA - Agua superficial detectada (red neuronal U-Net)", _texto_agua(agua)))
    filas += [
        ("Calidad de datos satelitales", gee.get("calidad", " - ")),
        ("NDVI en el punto óptimo", _fmt(gee.get("ndvi"), 3)),
        ("Humedad del suelo (ERA5) en el punto óptimo", _fmt(gee.get("humedad"), 4, " m³/m³")),
        ("Elevación en el punto óptimo", _fmt(gee.get("elevacion"), 1, " m")),
        ("Tramos de escorrentía/vaguada detectados", str(vaguadas.get("n_lineas", 0))),
        ("De ellos, vaguadas (cauce concentrado)", str(vaguadas.get("n_vaguadas", 0))),
        ("Acumulación máxima de flujo", _fmt(vaguadas.get("acum_maxima"), 1, " celdas")),
        ("Caudal PUNTA en tormenta (Método Racional; NO captable)", _fmt(vaguadas.get("caudal_maximo_ls"), 2, " L/s")),
        ("Geología del subsuelo", geologia.get("nombre_unidad", "Desconocido")),
        ("Valoración geológica", geologia.get("estado", " - ")),
    ]
    with pdf.table(col_widths=(95, 83), text_align=("LEFT", "LEFT")) as table:
        row = table.row()
        row.cell("Resultado", style=FontFace(emphasis="BOLD"))
        row.cell("Valor", style=FontFace(emphasis="BOLD"))
        for etiqueta, valor in filas:
            row = table.row()
            row.cell(etiqueta)
            row.cell(str(valor))


def _seccion_punto_optimo(pdf: InformePDF, gee: dict):
    pdf.add_page()
    pdf.titulo_seccion(2, "Punto óptimo BALSA  -  metodología")
    pdf.indicador_estrella(ESTRELLA_NEGRA, "Estrella NEGRA en el mapa  =  este punto (agua superficial)")
    pdf.parrafo(
        "El punto óptimo de BALSA se calcula combinando cuatro capas de datos satelitales/climáticos "
        "en una puntuación ponderada, calculada con Google Earth Engine sobre TODA la superficie de la "
        "parcela, y tomando el punto donde esa puntuación es máxima. Busca dónde se concentra agua "
        "SUPERFICIAL o casi-superficial -- útil para balsa o zona de recarga, no para un pozo profundo."
    )

    pdf.subtitulo("Variables consultadas")
    pdf.dato_consultado(
        "NDVI  -  Sentinel-2 (Copernicus), colección COPERNICUS/S2_SR_HARMONIZED, mediana de 2024 "
        "con filtro de nubosidad < 20%.",
        "NDVI = (B8 - B4) / (B8 + B4)   -   índice de vigor de la vegetación (infrarrojo cercano vs. rojo)."
    )
    pdf.dato_consultado(
        "Humedad del suelo  -  ERA5-Land (Copernicus/ECMWF), colección mensual, capa "
        "volumetric_soil_water_layer_1, media 2023-2024.",
        "Media aritmética mensual de humedad volumétrica del suelo (m³/m³) en el periodo."
    )
    pdf.dato_consultado(
        "Elevación y curvatura  -  SRTM 30 m (USGS/SRTMGL1_003).",
        "Curvatura = -Laplaciano(elevación)  -  mide si el terreno es cóncavo (acumula agua) o convexo."
    )
    pdf.subtitulo("Fórmula de la puntuación")
    pdf.parrafo(
        "Cada una de las 4 capas se normaliza entre 0 y 1 DENTRO de los límites de esta parcela "
        "(el valor más alto de la parcela = 1, el más bajo = 0). Después se combinan así:\n\n"
        "puntuación = NDVI_norm × 0.35  +  Humedad_norm × 0.30  +  (1 - Elevación_norm) × 0.20  "
        "+  Curvatura_norm × 0.15\n\n"
        "La elevación se invierte (1 - Elevación_norm) porque las zonas más bajas del relieve "
        "tienden a acumular agua. El resultado se suaviza con una mediana focal (radio 2 píxeles) "
        "para evitar puntos aislados por ruido del sensor, y se multiplica por 100 para expresarlo "
        "en porcentaje."
    )
    pdf.aviso(
        "Es una puntuación RELATIVA a esta parcela, no una probabilidad absoluta de encontrar agua. "
        "Un 33% en una parcela muy uniforme puede ser tan bueno como un 60% en otra muy heterogénea."
    )
    pdf.subtitulo("Resultado para esta parcela")
    pdf.parrafo(
        f"Punto óptimo BALSA: {gee.get('pozo_lat', 0):.6f}, {gee.get('pozo_lon', 0):.6f}   -   "
        f"Índice relativo (0-100, no es una probabilidad): {_fmt(gee.get('pozo_score'), 1)}\n"
        f"Calidad de datos: {gee.get('calidad', ' - ')} ({gee.get('calidad_txt', ' - ')})\n"
        f"NDVI en el punto: {_fmt(gee.get('ndvi'), 3)}   ·   "
        f"Humedad: {_fmt(gee.get('humedad'), 4)} m³/m³   ·   "
        f"Elevación: {_fmt(gee.get('elevacion'), 1)} m"
    )


def _seccion_punto_pozo(pdf: InformePDF, pozo: dict):
    pdf.add_page()
    pdf.titulo_seccion(3, "Punto óptimo POZO  -  metodología")
    pdf.indicador_estrella(ESTRELLA_DORADA, "Estrella DORADA en el mapa  =  este punto (agua subterránea)")

    if not pozo or not pozo.get("disponible"):
        pdf.parrafo(
            "No se ha podido calcular el punto óptimo de pozo para esta parcela "
            "(datos satelitales insuficientes en el periodo analizado)."
        )
        return

    pdf.parrafo(
        "El punto óptimo de POZO usa señales distintas a las de la balsa: no busca dónde se acumula "
        "agua superficial, sino dónde hay indicios de agua subterránea PERSISTENTE, refinados con "
        "pozos reales conocidos y la geología real del punto."
    )

    pdf.subtitulo("Variables consultadas")
    pdf.dato_consultado(
        "NDVI de verano (junio-agosto, 2022-2024)  -  Sentinel-2, mediana de las imágenes de esos meses "
        "con menos de un 30% de nubes (no la mediana anual que usa el punto de balsa).",
        "NDVI = (B8-B4)/(B8+B4)  -  vigor de la vegetación. Si se mantiene verde cuando todo se seca, "
        "puede indicar raíces que llegan a agua subterránea (o riego)."
    )
    pdf.dato_consultado(
        "Anomalía térmica de verano  -  Landsat 9 (o Landsat 8 si hay pocas imágenes), banda ST_B10, "
        "mediana de los veranos con menos de un 20% de nubes.",
        "Zonas más FRÍAS de lo normal en verano  -  la transpiración de vegetación con acceso a agua "
        "enfría la superficie."
    )
    pdf.dato_consultado(
        "Pozos reales cercanos  -  inventario de puntos de agua del IGME (mismo servicio que el "
        "resto de la app), en un radio de 20 km.",
        "Puntuación final = 80% puntuación satelital + 20% proximidad al pozo real más cercano (máxima encima del pozo, decreciente hasta 0 a 5 km); además, los candidatos "
        "se comparan entre sí con esa puntuación. Un pozo real cercano es la prueba más directa de acuífero explotable."
    )
    pdf.subtitulo("Fórmula de la puntuación satelital")
    pdf.parrafo(
        "Las dos capas se normalizan entre 0 y 1 dentro de la parcela y se combinan así:\n\n"
        "puntuación satelital = NDVI_verano_norm × 0.60  +  Frescor_verano_norm × 0.40\n\n"
        "Se suaviza con una mediana focal (radio 2 píxeles), se toman 200 puntos de la parcela a 30 m y "
        "se quedan los 15 mejores como candidatos. Después se aplica la puntuación final (80% satélite + "
        "20% cercanía a un pozo real) y, si la geología real (IGME GEODE) del ganador es desfavorable y el "
        "segundo candidato tiene al menos el 85% de su puntuación y mejor geología, se elige el segundo."
    )
    pdf.aviso(
        "Sigue sin ser una medición directa del acuífero -- ningún dato remoto sustituye un sondeo "
        "eléctrico vertical o una perforación real antes de decidir una inversión."
    )
    pdf.subtitulo("Resultado para esta parcela")
    dist_txt = (
        f"{pozo.get('distancia_pozo_real_m'):.0f} m" if pozo.get("distancia_pozo_real_m") is not None
        else "sin pozos reales cercanos usados"
    )
    geo_p = pozo.get("geologia", {})
    pdf.parrafo(
        f"Punto óptimo POZO: {pozo.get('pozo_profundo_lat', 0):.6f}, {pozo.get('pozo_profundo_lon', 0):.6f}   -   "
        f"Índice relativo final (0-100, no es una probabilidad): {_fmt(pozo.get('pozo_profundo_score'), 1)}\n"
        f"Índice relativo solo satelital (antes de refinar): {_fmt(pozo.get('score_satelite'), 1)}\n"
        f"Pozo real del IGME más cercano usado: {dist_txt}\n"
        f"Geología del punto: {geo_p.get('nombre_unidad', 'Desconocido')} ({geo_p.get('estado', ' - ')})"
    )


def _seccion_d8(pdf: InformePDF, vaguadas: dict):
    pdf.add_page()
    pdf.titulo_seccion(4, "Red de drenaje  -  vaguadas y escorrentías (algoritmo D8)")
    pdf.parrafo(
        "El algoritmo D8 es una técnica estándar de hidrología/SIG que simula por dónde correría el "
        "agua de lluvia según la forma del terreno  -  NO mide agua real, es un cálculo puramente "
        "topográfico."
    )
    pdf.dato_consultado(
        "Elevación  -  SRTM 90 m (OpenTopoData), malla de 20×20 puntos (400 en total) sobre el "
        "rectángulo que engloba la parcela.",
        "Se suaviza con un filtro de media móvil (tamaño 2) para reducir el ruido del sensor antes "
        "de calcular direcciones de flujo."
    )
    pdf.subtitulo("Paso 1  -  Dirección de flujo")
    pdf.parrafo(
        "Para cada una de las 400 celdas, se comparan sus 8 vecinas (N, NE, E, SE, S, SO, O, NO) y "
        "se elige aquella hacia la que la pendiente es más pronunciada cuesta abajo: "
        "pendiente = (altura_celda - altura_vecina) / distancia. Si ninguna vecina está más baja, la "
        "celda es una depresión y el agua se queda en ella."
    )
    pdf.subtitulo("Paso 2  -  Acumulación de flujo")
    pdf.parrafo(
        "Se recorren las celdas de mayor a menor altura (orden topológico) y cada celda suma a su "
        "vecina de destino su propia acumulación  -  así, una celda con acumulación = 50 significa que "
        "50 celdas de la malla terminan drenando hacia ahí según el relieve."
    )
    pdf.subtitulo("Paso 3  -  Clasificación en escorrentía / vaguada")
    pdf.parrafo(
        f"Una celda se considera parte de la red de drenaje si su acumulación >= {8} (umbral "
        f"STREAM_THR). Dentro de la red de drenaje, se distingue:\n"
        f"  -  Escorrentía (flujo difuso): acumulación entre 8 y 24 celdas.\n"
        f"  -  Vaguada (cauce concentrado): acumulación >= 25 celdas (umbral VAGUADA_THR).\n"
        "Ambos umbrales son heurísticos, ajustables, no una norma oficial."
    )
    pdf.subtitulo("Paso 4  -  Caudal PUNTA de cada tramo (Método Racional)")
    pdf.parrafo(
        "Q (m³/s) = C · I · A / 360,  con A en hectáreas e I en mm/h.\n"
        "El área de drenaje A se calcula multiplicando la acumulación de la celda por el área real "
        "que representa cada celda de la malla (calculada a partir del espaciado en metros). "
        "C = 0,30 (coeficiente de escorrentía típico de terreno agrícola/rural) e I = 20 mm/h "
        "(lluvia de referencia moderada-intensa) son valores de referencia ASUMIDOS, no medidos en "
        "esta parcela  -  el caudal resultante es orientativo."
    )
    pdf.aviso(
        "Este es el caudal PICO de escorrentía durante una tormenta de 20 mm/h, no un caudal captable ni "
        "sostenido. No indica cuánta agua se puede extraer de un pozo ni almacenar en una balsa; para eso "
        "ver el volumen anual al final de esta sección."
    )

    lineas = [f for f in vaguadas.get("geojson", {}).get("features", [])
              if f["properties"].get("tipo") == "linea_drenaje"]
    lineas_top = sorted(lineas, key=lambda f: f["properties"].get("caudal_ls", 0), reverse=True)[:15]

    if lineas_top:
        pdf.subtitulo(f"Tramos con mayor caudal punta estimado (mostrando {len(lineas_top)} de {len(lineas)} totales)")
        with pdf.table(col_widths=(30, 30, 30, 22, 26, 40),
                        text_align=("CENTER", "CENTER", "CENTER", "CENTER", "CENTER", "LEFT")) as table:
            row = table.row()
            for h in ["Tipo", "Latitud", "Longitud", "Acum.", "Área (ha)", "Caudal punta (L/s)"]:
                row.cell(h, style=FontFace(emphasis="BOLD"))
            for f in lineas_top:
                p = f["properties"]
                lon0, lat0 = f["geometry"]["coordinates"][0]
                row = table.row()
                row.cell("Vaguada" if p["categoria"] == "vaguada" else "Escorrentía")
                row.cell(f"{lat0:.5f}")
                row.cell(f"{lon0:.5f}")
                row.cell(_fmt(p.get("acumulacion"), 1))
                row.cell(_fmt(p.get("area_ha"), 3))
                row.cell(_fmt(p.get("caudal_ls"), 2))
    else:
        pdf.parrafo("No se ha detectado ningún tramo de drenaje por encima del umbral en esta parcela.")

    pdf.ln(2)
    pdf.subtitulo("Resumen de la red de drenaje")
    pdf.parrafo(
        f"Resolución de la malla: {vaguadas.get('resolucion', ' - ')}\n"
        f"Tramos totales: {vaguadas.get('n_lineas', 0)}  (de ellos, {vaguadas.get('n_vaguadas', 0)} son vaguadas)\n"
        f"Acumulación máxima registrada: {_fmt(vaguadas.get('acum_maxima'), 1)} celdas\n"
        f"Caudal PUNTA máximo (tormenta de referencia, no captable): {_fmt(vaguadas.get('caudal_maximo_ls'), 2)} L/s"
    )
    _bloque_volumen_anual(pdf, lineas)


def _seccion_puntos_optimos(pdf: InformePDF, vaguadas: dict, pozo: dict = None):
    pdf.add_page()
    pdf.titulo_seccion(5, "Puntos idóneos para balsa y para pozo")
    pdf.parrafo(
        "Esta sección reúne los sitios de la parcela más favorables para cada tipo de captación. Se "
        "calculan de forma distinta porque buscan cosas distintas: la balsa recoge el agua de lluvia que "
        "corre por la superficie, y el pozo busca agua bajo tierra."
    )

    # ── 5.1 Balsa: donde se junta la escorrentía (D8)
    pdf.subtitulo("5.1. Para BALSA (agua superficial): donde se junta la escorrentía")
    pdf.dato_consultado(
        "La misma malla de acumulación de flujo D8 calculada en la sección anterior.",
        "Se ordenan las 400 celdas de mayor a menor acumulación y se toman hasta 7, exigiendo que "
        "cada una esté separada de las ya elegidas al menos 2 celdas (para no agrupar varios puntos "
        "prácticamente en el mismo sitio)."
    )
    puntos = [f for f in vaguadas.get("geojson", {}).get("features", [])
              if f["properties"].get("tipo") == "punto_acumulacion"]
    puntos = sorted(puntos, key=lambda f: f["properties"].get("acumulacion", 0), reverse=True)
    if puntos:
        with pdf.table(
            col_widths=(14, 40, 40, 30, 54),
            text_align=("CENTER", "CENTER", "CENTER", "CENTER", "LEFT"),
        ) as table:
            row = table.row()
            for h in ["#", "Latitud", "Longitud", "Acumulación", "Interpretación"]:
                row.cell(h, style=FontFace(emphasis="BOLD"))
            for i, f in enumerate(puntos, start=1):
                p = f["properties"]
                lon0, lat0 = f["geometry"]["coordinates"]
                row = table.row()
                row.cell(str(i))
                row.cell(f"{lat0:.6f}")
                row.cell(f"{lon0:.6f}")
                row.cell(_fmt(p.get("acumulacion"), 1))
                etiqueta = p.get("etiqueta", "").replace("\U0001F4A7 ", "")
                row.cell(etiqueta)
        pdf.ln(2)
    else:
        pdf.parrafo("No se ha detectado ningún punto de acumulación significativo en esta parcela.")
    pdf.parrafo(
        "Además de estos puntos, la estrella negra (sección 2) marca la zona con mejores condiciones "
        "para acumular agua en superficie según la vegetación, la humedad y la forma del terreno."
    )

    # ── 5.2 Pozo: mejores puntos del índice de pozo
    pdf.subtitulo("5.2. Para POZO (agua subterránea): mejores puntos del índice de pozo")
    candidatos = (pozo or {}).get("candidatos_pozo") or []
    if not pozo or not pozo.get("disponible") or not candidatos:
        pdf.parrafo("No se han podido calcular puntos para pozo en esta parcela (datos satelitales insuficientes).")
        return
    pdf.dato_consultado(
        "Los candidatos del índice de pozo de la sección 3: vegetación verde y suelo fresco en verano "
        "(Sentinel-2 y Landsat), con un extra por cercanía a pozos reales del IGME.",
        "Se ordenan por la puntuación final y se toman hasta 5, separados al menos 40 m entre sí. El "
        "primero es la estrella dorada; los demás son alternativas por si el primero no es accesible."
    )
    with pdf.table(
        col_widths=(12, 34, 34, 30, 34, 34),
        text_align=("CENTER", "CENTER", "CENTER", "CENTER", "CENTER", "LEFT"),
    ) as table:
        row = table.row()
        for h in ["#", "Latitud", "Longitud", "Índice (0-100)", "Pozo IGME más cercano", "Nota"]:
            row.cell(h, style=FontFace(emphasis="BOLD"))
        for i, c in enumerate(candidatos, start=1):
            row = table.row()
            row.cell(str(i))
            row.cell(f"{c['lat']:.6f}")
            row.cell(f"{c['lon']:.6f}")
            row.cell(_fmt(c.get("score"), 1))
            d = c.get("distancia_pozo_real_m")
            row.cell(f"{d / 1000:.2f} km".replace(".", ",") if d is not None else "sin datos")
            row.cell("Estrella dorada" if c.get("principal") else "Alternativa")
    pdf.ln(2)
    pdf.aviso(
        "La profundidad estimada por la IA (sección 8) se calcula en la estrella dorada; dentro de una "
        "misma parcela suele ser parecida en todos estos puntos. La geología se comprueba en el punto "
        "principal y, a esta distancia, suele ser la misma."
    )


def _seccion_geologia(pdf: InformePDF, geologia: dict):
    pdf.add_page()
    pdf.titulo_seccion(6, "Geología del subsuelo")
    pdf.dato_consultado(
        "IGME  -  Mapa Geológico Continuo de España (GEODE), escala 1:50.000. Servicio oficial "
        "ArcGIS REST del Instituto Geológico y Minero de España, capa \"Recintos geología\", "
        "consultada en el centroide de la parcela. Si no hay cobertura en el punto, se usa "
        "Macrostrat (base de datos geológica global) como respaldo.",
        "Consulta espacial directa (sin cálculo numérico): se identifica en qué recinto geológico "
        "cartografiado cae el punto, y se clasifica el tipo de roca por palabras clave de su "
        "litología (caliza/arenisca/aluvial -> favorable; granito/esquisto/pizarra -> desfavorable; "
        "el resto -> intermedio) para estimar su aptitud como acuífero."
    )
    pdf.subtitulo("Resultado para esta parcela")
    pdf.parrafo(
        f"Unidad geológica: {geologia.get('nombre_unidad', 'Desconocido')}\n"
        f"Tipo de roca: {geologia.get('roca', 'Desconocido')}\n"
        f"Valoración: {geologia.get('estado', ' - ')}\n"
        f"Descripción: {geologia.get('descripcion', 'Sin descripción')}"
    )
    pdf.aviso(
        "La valoración se basa en palabras clave de la litología cartografiada. Las formaciones aluviales "
        "y terrazas cuaternarias (gravas y arenas) suelen ser permeables, y una valoración intermedia "
        "no implica que no pueda haber un buen acuífero: confirmar con pozos vecinos y geofísica."
    )


def _seccion_fuentes(pdf: InformePDF):
    pdf.add_page()
    pdf.titulo_seccion(11, "Fuentes de datos consultadas  -  resumen")
    fuentes = [
        ("Sentinel-2 (Copernicus/ESA)", "NDVI (vigor de la vegetación) y bandas de entrada de la red U-Net, resolución 10 m"),
        ("Landsat 8/9 (NASA/USGS)", "Temperatura de la superficie en verano (anomalía térmica), resolución 30 m"),
        ("ERA5-Land (Copernicus/ECMWF)", "Humedad volumétrica del suelo, resolución ~9 km"),
        ("SRTM (NASA/USGS)", "Elevación del terreno, resolución 30-90 m"),
        ("IGME  -  GEODE", "Mapa geológico continuo de España, escala 1:50.000"),
        ("Macrostrat", "Base de datos geológica global (respaldo fuera de cobertura IGME)"),
        ("IGME  -  Inventario de puntos de agua", "Pozos reales con profundidad registrada (entrenamiento del modelo y pozos cercanos)"),
        ("ESA WorldCover 2021", "Mapa de aguas permanentes usado para entrenar la red U-Net"),
        ("Modelo de profundidad (Gradient Boosting)", "Validación cruzada espacial por bloques de 10 km y por provincia (metricas.json)"),
        ("Red neuronal U-Net", "Detección de agua superficial, evaluada en zonas no usadas para entrenar (metricas_unet.json)"),
        ("Algoritmo D8 propio", "Dirección y acumulación de flujo hídrico sobre el relieve"),
        ("Método Racional (hidrología)", "Caudal punta de escorrentía a partir del área de drenaje"),
    ]
    with pdf.table(col_widths=(60, 118), text_align=("LEFT", "LEFT")) as table:
        row = table.row()
        row.cell("Fuente", style=FontFace(emphasis="BOLD"))
        row.cell("Qué aporta", style=FontFace(emphasis="BOLD"))
        for nombre, uso in fuentes:
            row = table.row()
            row.cell(nombre)
            row.cell(uso)

    pdf.ln(4)
    pdf.aviso(
        "Este informe explica el cálculo con datos remotos/satelitales, disponible sin coste ni "
        "desplazamiento. Ningún método remoto sustituye una prospección física en campo (sondeo "
        "eléctrico vertical, perforación con control geológico y ensayo de bombeo) antes de decidir "
        "una inversión en un pozo."
    )


import math
import json
from pathlib import Path


def _haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(a))


def _bloque_volumen_anual(pdf, lineas):
    pdf.ln(2)
    pdf.subtitulo("Volumen anual teórico de escorrentía (dato útil para dimensionar una balsa)")
    areas = [f["properties"].get("area_ha") for f in lineas
             if f["properties"].get("area_ha") is not None]
    if not areas:
        pdf.parrafo("No hay tramos con área de drenaje calculada; no se puede estimar el volumen.")
        return
    a_ha = max(areas)
    c = 0.30
    pdf.parrafo(
        "El caudal punta no indica cuánta agua se puede almacenar. Para una balsa importa el VOLUMEN "
        "anual: V (m³/año) = P (mm/año) × C × A (ha) × 10, con C = 0,30 y A = área de drenaje del "
        f"tramo principal ({a_ha:.1f} ha). La lluvia anual P NO se ha medido en esta parcela: la tabla "
        "muestra tres escenarios y debe sustituirse por la pluviometría de la estación AEMET más próxima."
    )
    with pdf.table(col_widths=(60, 118), text_align=("CENTER", "LEFT")) as table:
        row = table.row()
        row.cell("Lluvia anual P (escenario)", style=FontFace(emphasis="BOLD"))
        row.cell("Volumen teórico de escorrentía (m³/año)", style=FontFace(emphasis="BOLD"))
        for p_mm in (300, 500, 700):
            row = table.row()
            row.cell(f"{p_mm} mm/año")
            row.cell(f"{p_mm * c * a_ha * 10:,.0f}".replace(",", "."))
    pdf.ln(2)
    pdf.aviso(
        "Es un máximo teórico: el volumen realmente aprovechable es menor (infiltración, evaporación, "
        "desbordes y pérdidas de la balsa). Con un MDT de 90 m en terreno llano, el área de drenaje "
        "también es incierta."
    )


def _seccion_pozos_cercanos(pdf, coords, pozo):
    pdf.add_page()
    pdf.titulo_seccion(7, "Pozos reales cercanos y profundidad esperada")
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    lat0, lon0 = sum(lats) / len(lats), sum(lons) / len(lons)
    if pozo and pozo.get("disponible") and pozo.get("pozo_profundo_lat") is not None:
        lat0, lon0 = pozo["pozo_profundo_lat"], pozo["pozo_profundo_lon"]

    vivos = (pozo or {}).get("pozos_cercanos")
    if vivos:
        lista = [(q["distancia_m"], {"profundidad_m": q.get("profundidad")}) for q in vivos]
        fuente = ("Servicio en línea del inventario de puntos de agua del IGME: la misma consulta que usa el "
                  "cálculo del punto óptimo de pozo. Se muestran los pozos más cercanos al punto óptimo, con "
                  "o sin profundidad registrada.")
    else:
        try:
            from services.pozos_reales_service import _cargar_pozos
            pozos = [q for q in _cargar_pozos()
                     if q.get("profundidad_m") is not None and q.get("lat") is not None
                     and q.get("lon") is not None and q["profundidad_m"] > 0]
        except Exception:
            pozos = []
        lista = sorted(((_haversine_m(lat0, lon0, q["lat"], q["lon"]), q) for q in pozos),
                       key=lambda x: x[0])
        fuente = ("Inventario local de puntos de agua del IGME (copia descargada; puede diferir del servicio en "
                  "línea). Solo pozos con profundidad registrada.")
    pdf.dato_consultado(
        fuente,
        "Distancia en línea recta al punto óptimo de pozo. Profundidad esperada: media ponderada por el "
        "inverso del cuadrado de la distancia (IDW) de los 5 pozos más cercanos con profundidad "
        "registrada <= 200 m."
    )
    if not lista:
        pdf.parrafo("No se han podido consultar pozos reales para esta zona.")
        return
    pdf.subtitulo("Pozos reales más cercanos al punto óptimo")
    with pdf.table(col_widths=(60, 118), text_align=("CENTER", "CENTER")) as table:
        row = table.row()
        for h in ["Distancia (km)", "Profundidad registrada (m)"]:
            row.cell(h, style=FontFace(emphasis="BOLD"))
        for d, q in lista[:8]:
            row = table.row()
            row.cell(f"{d / 1000:.2f}")
            pr = q.get("profundidad_m")
            row.cell(_fmt(pr, 1) if pr is not None else "sin dato")
    vec = [(d, q) for d, q in lista
           if q.get("profundidad_m") is not None and 0 < q["profundidad_m"] <= 200][:5]
    pdf.ln(2)
    pdf.subtitulo("Profundidad esperada (orientativa)")
    d0, q0 = lista[0]
    pr0 = q0.get("profundidad_m")
    cerca = (f"Pozo real más cercano: a {d0:.0f} m, "
             + (f"con {pr0:.0f} m de profundidad." if pr0 is not None else "sin profundidad registrada."))
    if vec:
        pesos = [1.0 / (max(d, 1.0) ** 2) for d, _ in vec]
        est = sum(w * q["profundidad_m"] for w, (_, q) in zip(pesos, vec)) / sum(pesos)
        profs = [q["profundidad_m"] for _, q in vec]
        pdf.parrafo(
            f"Estimación IDW con los {len(vec)} pozos más cercanos con profundidad registrada: {est:.0f} m. "
            f"Rango observado en esos pozos: {min(profs):.0f} - {max(profs):.0f} m. {cerca}"
        )
        if vec[0][0] > 5000:
            pdf.aviso("El pozo con profundidad más cercano está a más de 5 km: la estimación es poco representativa de esta parcela.")
    else:
        pdf.parrafo(f"Ningún pozo cercano tiene profundidad registrada (<= 200 m); no se calcula estimación. {cerca}")
    pdf.aviso(
        "La profundidad del inventario es la de la CAPTACIÓN (perforación), no el nivel del agua. El rango "
        "no es una profundidad de perforación recomendada: el nivel freático debe medirse en pozos "
        "vecinos y confirmarse en campo."
    )
    pdf.parrafo(
        "Esta media sencilla es solo una referencia. La estimación del modelo de inteligencia artificial, "
        "con su margen de error, está en la sección 8."
    )


def _cargar_json(nombre):
    ruta = Path(__file__).resolve().parent.parent / "modelo" / nombre
    try:
        return json.loads(ruta.read_text(encoding="utf-8"))
    except Exception:
        return None


def _seccion_ia(pdf, ia):
    pdf.add_page()
    pdf.titulo_seccion(8, "Lo que ha calculado la inteligencia artificial")
    pdf.parrafo(
        "La aplicación incluye dos modelos entrenados con datos reales: uno estima a qué profundidad "
        "están los pozos en la zona y otro localiza el agua que se ve en superficie. Esta sección "
        "recoge lo que dicen para esta parcela; la sección 9 explica cuánto se puede confiar en ellos."
    )
    ia = ia or {}
    met = ia.get("metricas_profundidad") or _cargar_json("metricas.json") or {}

    # ── Profundidad (Gradient Boosting)
    pdf.subtitulo("8.1. Profundidad estimada del agua subterránea (Gradient Boosting)")
    prof = ia.get("profundidad") or {}
    n_pozos = met.get("n_pozos_entrenamiento")
    pdf.dato_consultado(
        f"Modelo de Gradient Boosting entrenado con {n_pozos if n_pozos else 'los'} pozos reales del IGME "
        "en Andalucía. Usa cuatro datos del punto: la profundidad de los 5 pozos conocidos más cercanos "
        "(media ponderada por la distancia), la distancia al más cercano, la elevación (SRTM 30 m) y la "
        "geología (IGME GEODE).",
        "El modelo aprende la profundidad a partir de esos cuatro datos. El margen de error (intervalo del "
        "80%) sale de los errores que cometió el modelo en zonas que no vio al entrenar, y se ensancha "
        "cuando no hay pozos conocidos cerca (predicción conformal)."
    )
    if prof.get("disponible"):
        sitio = prof.get("sitio", "punto analizado")
        punto = prof.get("punto") or {}
        pdf.parrafo(
            f"Punto analizado: {sitio} ({punto.get('lat', 0):.6f}, {punto.get('lon', 0):.6f})\n"
            f"Profundidad estimada: {_texto_profundidad(prof)}\n"
            f"Categoría: {prof.get('veredicto', '').split(':')[0]}  -  {prof.get('descripcion', '')}\n"
            f"Pozo conocido más cercano: a {_num(prof.get('distancia_pozo_cercano_km', 0), 2)} km"
        )
        vecinos = prof.get("pozos_vecinos") or []
        if vecinos:
            pdf.subtitulo("Pozos del IGME en los que se apoya la estimación")
            with pdf.table(col_widths=(60, 118), text_align=("CENTER", "CENTER")) as table:
                row = table.row()
                for h in ["Distancia (km)", "Profundidad registrada (m)"]:
                    row.cell(h, style=FontFace(emphasis="BOLD"))
                for v in vecinos:
                    row = table.row()
                    row.cell(_num(v.get("distancia_km", 0), 2))
                    row.cell(_num(v.get("profundidad_m", 0), 1))
            pdf.ln(2)
        pdf.parrafo(
            "Qué quiere decir: de cada 10 pozos situados en sitios parecidos, unos 8 tienen una profundidad "
            "dentro de ese rango. Si el rango es muy amplio, el modelo tiene poca información y hace falta un "
            "estudio de campo antes de decidir."
        )
        for aviso in prof.get("avisos") or []:
            pdf.aviso(aviso)
    else:
        pdf.parrafo(f"No disponible: {prof.get('mensaje', 'el modelo de profundidad no está entrenado.')}")

    # ── Agua superficial (U-Net)
    pdf.subtitulo("8.2. Agua superficial detectada (red neuronal U-Net)")
    agua = ia.get("agua") or {}
    pdf.dato_consultado(
        "Imágenes Sentinel-2 del último verano completo (mediana de junio a septiembre), 6 bandas a 10 m, "
        "sobre la parcela y 150 m alrededor.",
        "Una red neuronal convolucional (U-Net), entrenada con el mapa de aguas permanentes de ESA "
        "WorldCover, decide para cada píxel de 10 x 10 m si es agua. Los píxeles de agua contiguos forman "
        "masas de agua; se descartan las de menos de 4 píxeles (unos 400 m²)."
    )
    if agua.get("disponible"):
        pdf.parrafo(
            f"Periodo de las imágenes: {agua.get('periodo', ' - ')}\n"
            f"Agua dentro de la parcela: {_num(agua.get('area_agua_parcela_ha', 0), 2)} ha\n"
            f"Agua en la parcela y 150 m alrededor: {_num(agua.get('area_agua_ha', 0), 2)} ha, en "
            f"{agua.get('n_masas', 0)} masa(s) de agua"
        )
        masas = agua.get("masas") or []
        if masas:
            with pdf.table(col_widths=(14, 34, 40, 40, 50),
                           text_align=("CENTER", "CENTER", "CENTER", "CENTER", "CENTER")) as table:
                row = table.row()
                for h in ["#", "Superficie (ha)", "Latitud", "Longitud", "Dentro de la parcela"]:
                    row.cell(h, style=FontFace(emphasis="BOLD"))
                for i, m in enumerate(masas, start=1):
                    row = table.row()
                    row.cell(str(i))
                    row.cell(_num(m.get("area_ha", 0), 3))
                    row.cell(f"{m.get('lat', 0):.6f}")
                    row.cell(f"{m.get('lon', 0):.6f}")
                    row.cell("Sí" if m.get("en_parcela") else "No")
            pdf.ln(2)
        else:
            pdf.parrafo("No se ha detectado agua superficial en la parcela ni a menos de 150 m.")
        if agua.get("aviso"):
            pdf.aviso(agua["aviso"])
        pdf.aviso(
            "Detecta el agua que se ve desde el satélite en verano. Las balsas muy pequeñas (menos de unos "
            "400 m²), cubiertas o vacías en esas fechas pueden no aparecer."
        )
    else:
        pdf.parrafo(f"No disponible: {agua.get('mensaje', 'la red U-Net no está entrenada.')}")


def _seccion_fiabilidad_modelo(pdf, ia=None):
    pdf.add_page()
    pdf.titulo_seccion(9, "Fiabilidad de los modelos y limitaciones")
    ia = ia or {}
    m = ia.get("metricas_profundidad") or _cargar_json("metricas.json")
    u = ia.get("metricas_unet") or _cargar_json("metricas_unet.json")

    pdf.subtitulo("Modelo de profundidad (Gradient Boosting)")
    try:
        comp = m["comparacion"]
        gb = comp["Gradient Boosting"]
        idw = next(v for k, v in comp.items() if k.startswith("IDW"))
        prov = (m.get("escenario_provincia_no_vista") or {}).get("Gradient Boosting")
        cob = m["intervalo"]["cobertura"]
        lineas = (
            f"Se entrenó con {m['n_pozos_entrenamiento']} pozos reales del IGME y se evaluó con validación "
            "cruzada espacial: los pozos se agrupan en bloques de 10 x 10 km y el modelo se mide siempre en "
            "bloques que no ha visto al entrenar. Resultados:\n"
            f"  -  Error medio: {_num(gb['mae_m'])} m (usando solo la media de los pozos vecinos: "
            f"{_num(idw['mae_m'])} m).\n"
            f"  -  Error mediano: {_num(gb['mediana_error_m'])} m: en la mitad de los pozos se equivoca menos.\n"
            f"  -  R² = {_num(gb['r2'], 2)}: positivo pero muy bajo; el modelo explica una parte pequeña de "
            "las diferencias de profundidad entre pozos.\n"
        )
        if prov:
            lineas += (f"  -  En una provincia sin ningún pozo conocido: R² = {_num(prov['r2'], 2)}; ahí no "
                       "mejora a dar siempre la profundidad típica.\n")
        lineas += f"  -  El intervalo del 80% contuvo la profundidad real en el {_num(cob * 100)}% de los pozos de validación."
        pdf.parrafo(lineas)
        pdf.aviso(
            "Por eso la profundidad estimada es orientativa: sirve para saber si en la zona los pozos suelen "
            "ser someros o profundos, no para presupuestar un sondeo."
        )
    except Exception:
        pdf.parrafo("No se han podido cargar las métricas de validación del modelo de profundidad.")

    pdf.subtitulo("Red neuronal de agua superficial (U-Net)")
    try:
        t = u["test"]
        un = t["U-Net (CNN)"]
        ndwi = next(v for k, v in t.items() if k.startswith("NDWI"))
        pdf.parrafo(
            f"Se evaluó en {u['n_parches']['test']} recortes de imagen de zonas no usadas para entrenar:\n"
            f"  -  IoU = {_num(un['iou'], 2)}: grado de coincidencia con el mapa de referencia (1 sería "
            f"coincidencia perfecta), frente a {_num(ndwi['iou'], 2)} del índice clásico NDWI.\n"
            f"  -  Precisión = {_num(un['precision'], 2)}: casi todo lo que marca como agua lo es.\n"
            f"  -  Sensibilidad = {_num(un['recall'], 2)}: detecta unos {round(un['recall'] * 10)} de cada 10 "
            "píxeles de agua."
        )
    except Exception:
        pdf.parrafo("No se han podido cargar las métricas de la red U-Net.")

    pdf.subtitulo("Otras limitaciones del análisis")
    pdf.parrafo(
        "-  Resolución del relieve: la red de drenaje usa una malla de 20x20 puntos sobre SRTM de 90 m. "
        "En terreno llano el error vertical del sensor puede ser comparable al relieve real, por lo que "
        "las vaguadas son indicativas. Para diseñar obras se recomienda un MDT de mayor resolución "
        "(MDT05 o LiDAR del PNOA, CNIG).\n"
        "-  La humedad del suelo (ERA5-Land, ~9 km) es casi constante dentro de una parcela.\n"
        "-  El NDVI puede reflejar riego o cultivo y no necesariamente agua subterránea.\n"
        "-  Los índices relativos solo comparan puntos de esta parcela entre sí.\n"
        "-  No hay validación de campo (sondeos, piezometría) de los puntos propuestos."
    )


def _seccion_campo_y_legal(pdf):
    pdf.add_page()
    pdf.titulo_seccion(10, "Recomendaciones de campo y marco legal")
    pdf.subtitulo("Antes de perforar")
    pdf.parrafo(
        "-  Visitar pozos vecinos: medir nivel del agua, profundidad y caudal real.\n"
        "-  Prospección geofísica (sondeo eléctrico vertical o tomografía eléctrica) en el punto propuesto.\n"
        "-  Ensayo de bombeo tras la perforación y análisis de calidad (conductividad/salinidad y nitratos).\n"
        "-  Comprobar si la zona es vulnerable a nitratos o si la masa de agua subterránea está en riesgo."
    )
    pdf.subtitulo("Marco legal (orientativo)")
    pdf.parrafo(
        "-  Según el art. 54.2 del Texto Refundido de la Ley de Aguas, el propietario puede aprovechar las "
        "aguas subterráneas de su finca sin concesión hasta 7.000 m³/año, pero el pozo debe inscribirse "
        "en el Registro de Aguas de la Confederación Hidrográfica.\n"
        "-  Por encima de ese volumen se necesita concesión, y en acuíferos sobreexplotados puede denegarse.\n"
        "-  Una balsa requiere autorización de la Confederación Hidrográfica y, según su tamaño, "
        "cumplir la normativa de seguridad. Consulte con el organismo de cuenca competente."
    )
    pdf.aviso(
        "Este informe es un CRIBADO PRELIMINAR por teledetección. No sustituye un estudio hidrogeológico "
        "firmado por un técnico competente ni constituye asesoramiento legal."
    )


def generar_informe_pdf(coords: list, resultado_gee: dict, resultado_vaguadas: dict, geologia: dict,
                        resultado_pozo: dict = None, resultado_ia: dict = None) -> bytes:
    """Genera el informe completo y devuelve los bytes del PDF."""
    pdf = InformePDF()
    fecha_txt = datetime.now().strftime("%d/%m/%Y %H:%M")

    _portada(pdf, coords, fecha_txt)
    pdf.ln(8)
    pdf.aviso("CRIBADO PRELIMINAR por teledetección: orienta dónde mirar primero, no garantiza la presencia de agua ni sustituye un estudio hidrogeológico.")
    _seccion_resumen(pdf, resultado_gee, resultado_vaguadas, geologia, resultado_pozo, resultado_ia)
    _seccion_punto_optimo(pdf, resultado_gee)
    _seccion_punto_pozo(pdf, resultado_pozo)
    _seccion_d8(pdf, resultado_vaguadas)
    _seccion_puntos_optimos(pdf, resultado_vaguadas, resultado_pozo)
    _seccion_geologia(pdf, geologia)
    _seccion_pozos_cercanos(pdf, coords, resultado_pozo)
    _seccion_ia(pdf, resultado_ia)
    _seccion_fiabilidad_modelo(pdf, resultado_ia)
    _seccion_campo_y_legal(pdf)
    _seccion_fuentes(pdf)

    return bytes(pdf.output())
