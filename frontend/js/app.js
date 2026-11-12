/* ──────────────────────────────────────────────────────────────────
   app.js — Buscador de Agua Subterránea v4.4
   Con autenticación JWT + roles usuario/admin
   v4: profundidad con intervalo de error (Gradient Boosting) y agua
   superficial con una red U-Net (CNN)
   ────────────────────────────────────────────────────────────────── */

const API_BASE = "/api";

// ── Sesión ────────────────────────────────────────────────────────
const SESSION = {
  token:    localStorage.getItem("token"),
  role:     localStorage.getItem("role"),
  username: localStorage.getItem("username"),
  email:    localStorage.getItem("email"),
};

// Si no hay token, redirigir al login
if (!SESSION.token) {
  window.location.href = "login.html";
}

// Helper para añadir Authorization header a todas las peticiones
async function apiFetch(endpoint, options = {}) {
  const headers = {
    "Content-Type": "application/json",
    "Authorization": `Bearer ${SESSION.token}`,
    ...(options.headers || {}),
  };

  const res = await fetch(`${API_BASE}${endpoint}`, { ...options, headers });

  // Token expirado o inválido
  if (res.status === 401) {
    localStorage.clear();
    window.location.href = "login.html";
    return;
  }

  return res;
}

async function apiFetchForm(endpoint, formData) {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    method: "POST",
    headers: { "Authorization": `Bearer ${SESSION.token}` },
    body: formData,
  });
  if (res.status === 401) { localStorage.clear(); window.location.href = "login.html"; return; }
  return res;
}

// Tareas largas (informe PDF, NDVI de 4 años): el servidor las lanza en
// segundo plano y aquí se pregunta cada 3 s si han terminado, para que
// ninguna petición dure tanto como para cortarse por el camino.
async function esperarTrabajo(ruta, cuerpo, esPdf = false) {
  const ini = await apiFetch(ruta, { method: "POST", body: JSON.stringify(cuerpo) });
  if (!ini?.ok) throw new Error(ini ? await ini.text() : "Sesión caducada");
  const { id } = await ini.json();
  const limite = Date.now() + 10 * 60 * 1000;
  while (Date.now() < limite) {
    await new Promise(r => setTimeout(r, 3000));
    const r = await apiFetch(`/trabajos/${id}`);
    if (!r?.ok) throw new Error(r ? await r.text() : "Sesión caducada");
    const e = await r.json();
    if (e.estado === "error") throw new Error(e.error);
    if (e.estado === "ok") return esPdf ? await apiFetch(`/trabajos/${id}/pdf`) : e.resultado;
  }
  throw new Error("El servidor está tardando demasiado. Inténtalo de nuevo en unos minutos.");
}

// ── Estado global ────────────────────────────────────────────────
const state = {
  mapa:            null,
  drawnItems:      null,
  markerPrincipal: null,
  layerIA:         null,
  layerVaguadas:   null,
  layerThermal:    null,
  layerAgua:       null,
  layerMediciones: null,
  ultimasParcelas: null,
  graficaNDVI:     null,
};

// ── Helpers UI ───────────────────────────────────────────────────
function showSpinner(msg = "Cargando...") {
  document.getElementById("spinner-msg").textContent = msg;
  document.getElementById("spinner").classList.remove("hidden");
}
function hideSpinner() {
  document.getElementById("spinner").classList.add("hidden");
}
function metricCard(label, value) {
  return `<div class="metric-card"><div class="metric-label">${label}</div><div class="metric-value">${value}</div></div>`;
}
function alertHTML(type, msg) {
  return `<div class="alert alert-${type} mt-2">${msg}</div>`;
}
function showBanner(msg, type = "danger") {
  const b = document.createElement("div");
  b.className = `alert alert-${type} fixed bottom-4 right-4 z-50 max-w-xs shadow-lg`;
  b.innerHTML = msg;
  document.body.appendChild(b);
  setTimeout(() => b.remove(), 6000);
}

// ── Inicialización del mapa ──────────────────────────────────────
function initMapa() {
  const lat  = parseFloat(document.getElementById("latitud").value);
  const lon  = parseFloat(document.getElementById("longitud").value);
  const zoom = parseInt(document.getElementById("zoom").value);

  state.mapa = L.map("map", { center: [lat, lon], zoom });

  L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    { attribution: "Esri World Imagery", maxZoom: 19 }
  ).addTo(state.mapa);

  state.drawnItems    = new L.FeatureGroup().addTo(state.mapa);
  state.layerIA       = new L.FeatureGroup().addTo(state.mapa);
  state.layerVaguadas = new L.FeatureGroup().addTo(state.mapa);
  state.layerThermal  = new L.FeatureGroup().addTo(state.mapa);
  state.layerAgua     = new L.FeatureGroup().addTo(state.mapa);
  // Capa de mediciones (distancias y perpendiculares) — independiente de la
  // parcela; las líneas se quedan guardadas en el mapa hasta que el usuario
  // pulse "Borrar todo".
  state.layerMediciones = new L.FeatureGroup().addTo(state.mapa);

  // Capa SIGPAC — recintos (planimetría oficial de parcelas agrícolas, FEGA)
  // Se añade como capa opcional (desactivada por defecto) mediante el
  // control de capas de Leaflet, sin alterar ninguna capa existente.
  state.layerSigpac = L.tileLayer.wms("https://sigpac-hubcloud.es/wms", {
    layers: "AU.Sigpac:recinto",
    format: "image/png",
    transparent: true,
    version: "1.3.0",
    attribution: "SIGPAC © FEGA",
    maxZoom: 21,
  });

  // Capa IGN — Curvas de nivel (equidistancia 50 m). Nombre de capa y
  // estilo confirmados directamente en el GetCapabilities oficial que
  // compartiste: EL.ContourLine, cobertura hasta escala 1:200.000. Al ser
  // solo líneas (no un raster de área completa), no tiene el problema de
  // "se ve todo negro" del Modelo Digital del Terreno en zonas bajas.
  state.layerRelieveIGN = L.tileLayer.wms("https://servicios.idee.es/wms-inspire/mdt", {
    layers: "EL.ContourLine",
    format: "image/png",
    transparent: true,
    version: "1.3.0",
    attribution: "IGN — Curvas de nivel",
    maxZoom: 19,
  });

  // Pozos reales del IGME — dataset LOCAL (procesado de tus Excel), no una
  // consulta a un servidor externo, así que carga rápido y siempre está
  // disponible. Se piden los que caen dentro de la vista actual del mapa,
  // y se recargan solos al mover o hacer zoom mientras la capa esté activa.
  state.layerPozosReales = L.layerGroup();

  async function cargarPozosRealesEnVista() {
    const b = state.mapa.getBounds();
    try {
      const res = await apiFetch("/pozos-reales-area", {
        method: "POST",
        body: JSON.stringify({
          lat_min: b.getSouth(), lat_max: b.getNorth(),
          lon_min: b.getWest(),  lon_max: b.getEast(),
        }),
      });
      if (!res.ok) return;
      const d = await res.json();
      state.layerPozosReales.clearLayers();
      d.pozos.forEach((p) => {
        L.circleMarker([p.lat, p.lon], {
          radius: 4, color: "#0ea5e9", weight: 1.5, fillColor: "#38bdf8", fillOpacity: 0.85,
        }).bindTooltip(
          `📍 Pozo real IGME${p.naturaleza ? " — " + p.naturaleza : ""}<br>` +
          (p.profundidad_m !== null && p.profundidad_m !== undefined ? `Profundidad: <strong>${p.profundidad_m} m</strong><br>` : "Profundidad: no registrada<br>") +
          `${p.municipio || ""}${p.municipio && p.provincia ? ", " : ""}${p.provincia || ""}`
        ).addTo(state.layerPozosReales);
      });
    } catch (e) {
      console.warn("No se pudieron cargar los pozos reales:", e);
    }
  }

  state.mapa.on("overlayadd", (e) => {
    if (e.layer === state.layerPozosReales) {
      cargarPozosRealesEnVista();
      state.mapa.on("moveend", cargarPozosRealesEnVista);
    }
  });
  state.mapa.on("overlayremove", (e) => {
    if (e.layer === state.layerPozosReales) {
      state.mapa.off("moveend", cargarPozosRealesEnVista);
      state.layerPozosReales.clearLayers();
    }
  });

  L.control.layers(
    null,
    {
      "🟨 Recintos SIGPAC": state.layerSigpac,
      "🗻 Curvas de nivel (IGN)": state.layerRelieveIGN,
      "📍 Pozos reales IGME (Andalucía)": state.layerPozosReales,
    },
    { position: "topright", collapsed: false }
  ).addTo(state.mapa);

  const drawControl = new L.Control.Draw({
    position: "topleft",
    draw: {
      polyline: false, circle: false, circlemarker: false, marker: true,
      polygon:  { shapeOptions: { color: "#3b82f6", fillOpacity: 0.1 } },
      rectangle:{ shapeOptions: { color: "#3b82f6", fillOpacity: 0.1 } },
    },
    edit: { featureGroup: state.drawnItems },
  });
  state.mapa.addControl(drawControl);

  state.mapa.on(L.Draw.Event.CREATED, (e) => {
    state.drawnItems.clearLayers();
    state.drawnItems.addLayer(e.layer);
    const geom = e.layer.toGeoJSON().geometry;
    if (geom.type === "Polygon") {
      state.ultimasParcelas = geom.coordinates[0];
      activarBotonesParcela();
    } else if (geom.type === "Point") {
      document.getElementById("latitud").value  = geom.coordinates[1].toFixed(6);
      document.getElementById("longitud").value = geom.coordinates[0].toFixed(6);
    }
  });

  inicializarHerramientaMedicion();

  colocarMarcadorPrincipal(lat, lon, "blue");
}

function colocarMarcadorPrincipal(lat, lon, color) {
  if (state.markerPrincipal) state.mapa.removeLayer(state.markerPrincipal);
  const emoji = color === "green" ? "🟢" : color === "orange" ? "🟠" : "🔵";
  state.markerPrincipal = L.marker([lat, lon], {
    draggable: true,
    icon: L.divIcon({ className: "", html: `<div style="font-size:24px;line-height:1;cursor:grab;">${emoji}</div>`, iconAnchor: [12, 12] }),
  }).addTo(state.mapa).bindTooltip("Centro de tu finca — arrástralo para moverlo");

  // Arrastrar el marcador actualiza las coordenadas del formulario.
  state.markerPrincipal.on("dragend", (e) => {
    const pos = e.target.getLatLng();
    document.getElementById("latitud").value  = pos.lat.toFixed(6);
    document.getElementById("longitud").value = pos.lng.toFixed(6);
  });
}

// ── Colocar el localizador con un clic en el mapa ────────────────
let colocandoEnMapa = false;

function clickColocarMarcador(e) {
  const { lat, lng } = e.latlng;
  document.getElementById("latitud").value  = lat.toFixed(6);
  document.getElementById("longitud").value = lng.toFixed(6);
  colocarMarcadorPrincipal(lat, lng, "blue");
  finalizarColocarEnMapa();
}

function finalizarColocarEnMapa() {
  state.mapa.off("click", clickColocarMarcador);
  colocandoEnMapa = false;
  const btn = document.getElementById("btn-clic-mapa");
  btn.className = "btn-secondary mt-2 w-full";
  btn.textContent = "🎯 Colocar en el mapa";
}

document.getElementById("btn-clic-mapa").addEventListener("click", () => {
  if (colocandoEnMapa) { finalizarColocarEnMapa(); return; }
  colocandoEnMapa = true;
  const btn = document.getElementById("btn-clic-mapa");
  btn.className = "btn-primary mt-2 w-full";
  btn.textContent = "❌ Cancelar";
  showBanner("Haz clic en cualquier punto del mapa para situar el localizador ahí.", "info");
  state.mapa.on("click", clickColocarMarcador);
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && colocandoEnMapa) finalizarColocarEnMapa();
});

function activarBotonesParcela() {
  [["btn-analizar-parcela","🤖 Analizar parcela con IA"],
   ["btn-vaguadas","🌊 Ver flujo de agua y vaguadas"],
   ["btn-ndvi-serie","📈 Ver NDVI 4 años"],
   ["btn-thermal","🌡️ Ver anomalías térmicas"],
   ["btn-agua-superficial","💧 Detectar agua superficial (CNN)"],
   ["btn-informe-pdf","📄 Descargar informe PDF completo"]].forEach(([id, txt]) => {
    const b = document.getElementById(id);
    if (b) { b.disabled = false; b.textContent = txt; }
  });
  document.getElementById("parcela-info").classList.remove("hidden");
  limpiarResultadoPuntos();
}

// ── DESCARGAR INFORME PDF ──────────────────────────────────────────
// Recalcula el análisis completo (punto óptimo, vaguadas, geología) en
// el backend y genera un PDF explicando, para cada resultado, qué fuente
// se consultó y qué cálculo matemático se aplicó. Es una llamada aparte
// (no reutiliza los resultados ya mostrados en pantalla), así que puede
// tardar tanto como "Analizar parcela con IA" + "Ver vaguadas" juntos.
document.getElementById("btn-informe-pdf").addEventListener("click", async () => {
  if (!state.ultimasParcelas) { showBanner("Dibuja primero un polígono.", "warning"); return; }
  showSpinner("Generando informe PDF... recalcula el análisis completo, puede tardar un poco");
  try {
    const res = await esperarTrabajo("/trabajos/informe-pdf", { coords: state.ultimasParcelas }, true);
    if (!res?.ok) throw new Error(res ? await res.text() : "Sesión caducada");
    const blob = await res.blob();
    const url  = window.URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href = url;
    a.download = "informe_agua_subterranea.pdf";
    document.body.appendChild(a);
    a.click();
    a.remove();
    // por si el navegador integrado no descarga solo, se deja un enlace un rato
    const aviso = document.createElement("div");
    aviso.className = "alert alert-success fixed bottom-4 right-4 z-50 max-w-xs shadow-lg";
    aviso.innerHTML = `✅ Informe listo. Si no se ha descargado, <a href="${url}" target="_blank" download="informe_agua_subterranea.pdf" style="text-decoration:underline">pulsa aquí</a>.`;
    document.body.appendChild(aviso);
    setTimeout(() => { aviso.remove(); window.URL.revokeObjectURL(url); }, 60000);
  } catch (err) {
    showBanner(`❌ Error generando el informe: ${err.message}`);
  } finally {
    hideSpinner();
  }
});

// ── Slider zoom ──────────────────────────────────────────────────
document.getElementById("zoom").addEventListener("input", (e) => {
  document.getElementById("zoom-val").textContent = e.target.value;
  if (state.mapa) state.mapa.setZoom(parseInt(e.target.value));
});

// ── Botón limpiar ────────────────────────────────────────────────
document.getElementById("btn-limpiar").addEventListener("click", () => {
  state.layerIA.clearLayers();
  state.layerVaguadas.clearLayers();
  state.layerThermal.clearLayers();
  state.layerAgua?.clearLayers();
  state.drawnItems.clearLayers();
  state.ultimasParcelas = null;
  if (state.graficaNDVI) { state.graficaNDVI.destroy(); state.graficaNDVI = null; }
  limpiarResultadoPuntos();
  ["card-ia","card-vaguadas","card-ndvi","card-thermal","card-agua","card-prediccion-modelo","leyenda-ia","leyenda-vaguadas","parcela-info","btn-limpiar"].forEach(id => {
    document.getElementById(id)?.classList.add("hidden");
  });
  ["btn-analizar-parcela","btn-vaguadas","btn-ndvi-serie","btn-thermal","btn-agua-superficial"].forEach(id => {
    const b = document.getElementById(id);
    if (b) { b.disabled = true; b.textContent = "Dibuja una parcela primero"; }
  });
});

// ── BUSCADOR ─────────────────────────────────────────────────────
async function buscarLugar(query) {
  if (!query.trim()) return;
  const resDiv = document.getElementById("buscador-resultados");
  resDiv.innerHTML = `<div class="text-white/50 text-xs py-1">Buscando...</div>`;
  try {
    const res = await fetch(`https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(query)}&format=json&limit=5&countrycodes=es`, { headers: { "Accept-Language": "es" } });
    const resultados = await res.json();
    if (!resultados.length) { resDiv.innerHTML = `<div class="text-white/50 text-xs py-1">No encontrado.</div>`; return; }
    resDiv.innerHTML = resultados.map(r => {
      const nombre = r.display_name.length > 60 ? r.display_name.substring(0, 60) + "..." : r.display_name;
      return `<div class="resultado-busqueda" data-lat="${r.lat}" data-lon="${r.lon}" data-nombre="${r.display_name.replace(/"/g,"'")}">📍 ${nombre}</div>`;
    }).join("");
    resDiv.querySelectorAll(".resultado-busqueda").forEach(el => {
      el.addEventListener("click", () => {
        const lat = parseFloat(el.dataset.lat), lon = parseFloat(el.dataset.lon);
        document.getElementById("latitud").value  = lat.toFixed(6);
        document.getElementById("longitud").value = lon.toFixed(6);
        state.mapa.setView([lat, lon], 15);
        colocarMarcadorPrincipal(lat, lon, "blue");
        resDiv.innerHTML = "";
        document.getElementById("buscador").value = el.dataset.nombre || "";
      });
    });
  } catch { resDiv.innerHTML = `<div class="text-white/50 text-xs py-1">Error al buscar.</div>`; }
}

document.getElementById("btn-buscar").addEventListener("click", () => {
  const query = document.getElementById("buscador").value.trim();
  const lat = parseFloat(document.getElementById("latitud").value);
  const lon = parseFloat(document.getElementById("longitud").value);
  if (query) { buscarLugar(query); return; }
  if (!isNaN(lat) && !isNaN(lon)) { state.mapa.setView([lat, lon], 15); colocarMarcadorPrincipal(lat, lon, "blue"); }
});
document.getElementById("buscador").addEventListener("keydown", e => { if (e.key === "Enter") buscarLugar(e.target.value); });

// ── IR A COORDENADAS ─────────────────────────────────────────────
document.getElementById("btn-ir-coordenadas").addEventListener("click", () => {
  const lat = parseFloat(document.getElementById("latitud").value);
  const lon = parseFloat(document.getElementById("longitud").value);
  const zoom = parseInt(document.getElementById("zoom").value);
  if (isNaN(lat) || isNaN(lon)) { showBanner("Coordenadas inválidas.", "warning"); return; }
  state.mapa.setView([lat, lon], zoom);
  colocarMarcadorPrincipal(lat, lon, "blue");
});

// ── VER HUMEDAD Y ELEVACIÓN ──────────────────────────────────────
document.getElementById("btn-humedad-elevacion").addEventListener("click", async () => {
  const lat = parseFloat(document.getElementById("latitud").value);
  const lon = parseFloat(document.getElementById("longitud").value);
  if (isNaN(lat) || isNaN(lon)) { showBanner("Introduce coordenadas válidas.", "warning"); return; }
  showSpinner("Consultando humedad, elevación y profundidad del agua...");
  try {
    const res = await apiFetch("/analizar-punto", { method: "POST", body: JSON.stringify({ latitud: lat, longitud: lon }) });
    if (!res.ok) throw new Error(await res.text());
    const d = await res.json();
    const p = d.profundidad || {};

    let tarjetasProfundidad = "";
    if (p.profundidad_freatico_m !== null && p.profundidad_freatico_m !== undefined) {
      tarjetasProfundidad =
        metricCard("💧 Nivel freático est.", `${p.profundidad_freatico_m} m`) +
        metricCard("⛏️ Excavar hasta (reserva)", `${p.profundidad_total_min_m}–${p.profundidad_total_max_m} m`);
    } else {
      tarjetasProfundidad = metricCard("💧 Nivel freático est.", "Sin pozos reales cerca");
    }

    document.getElementById("metricas-punto").innerHTML =
      metricCard("Humedad suelo", `${d.humedad.toFixed(4)} m³/m³`) +
      metricCard("Elevación", `${d.elevacion} m`) +
      tarjetasProfundidad;

    const CONFIANZA_TXT = { alta: "🟢 Alta", media: "🟡 Media", baja: "🟠 Baja", sin_datos_reales: "⚪ Sin pozos reales cerca" };
    let detalle = `<p><strong>Método:</strong> ${p.metodo || "—"}</p>`;
    if (p.pozo_mas_cercano_km !== null && p.pozo_mas_cercano_km !== undefined) {
      detalle += `<p class="mt-1"><strong>Confianza:</strong> ${CONFIANZA_TXT[p.confianza] || p.confianza} ` +
                 `(pozo real más cercano a ${p.pozo_mas_cercano_km} km)</p>`;
    }
    detalle += `<p class="mt-1"><strong>Tipo de acuífero estimado:</strong> ${p.tipo_acuifero_estimado || "—"} (geología: ${p.geologia || "Desconocida"})</p>`;
    if (p.margen_recomendado_min_m !== undefined) {
      detalle += `<p class="mt-1">Se recomienda excavar entre <strong>${p.margen_recomendado_min_m} y ${p.margen_recomendado_max_m} m</strong> ` +
                 `por debajo del nivel freático estimado, para absorber el descenso estacional y el abatimiento por bombeo — ` +
                 `no es una norma técnica oficial, es una referencia orientativa según el tipo de acuífero.</p>`;
    }
    if (p.pozos_referencia && p.pozos_referencia.length > 0) {
      detalle += `<p class="mt-2 text-white/50">Pozos reales usados (IGME): ` +
        p.pozos_referencia.map(x => `${x.toponimia || "s/n"} — ${x.profundidad_m} m (a ${x.distancia_km} km)`).join(" · ") +
        `</p>`;
    }
    document.getElementById("detalle-profundidad").innerHTML = detalle;
    document.getElementById("detalle-profundidad").classList.remove("hidden");

    document.getElementById("card-humedad-elevacion").classList.remove("hidden");
    anadirProfundidadModeloPunto(lat, lon);
  } catch (err) { showBanner(`❌ Error: ${err.message}`); }
  finally { hideSpinner(); }
});

// ── HERRAMIENTA DE MEDICIÓN (estilo SIGPAC: clic para medir) ────
// 📏 Medir distancia — clic para ir añadiendo puntos; cada línea se queda
//    guardada en el mapa con su distancia, y se puede seguir midiendo más
//    líneas sin perder las anteriores (igual que "Medir longitudes" en
//    SIGPAC: doble clic fija la medida y deja la herramienta lista para
//    la siguiente; Escape cancela la que está en curso).
// ⟂ Perpendicular — clic sobre una línea ya guardada para usarla de
//    referencia (se toma el TRAMO exacto donde se hace clic, no toda la
//    línea), clic para fijar el punto base sobre ese tramo, y mueve el
//    ratón + clic para fijar el otro extremo (siempre a 90° en vivo).
// 📍 Desde coordenadas — usa las coordenadas ya escritas arriba como
//    primer punto, y sigue igual que "Medir distancia" desde ahí.
// 🗑️ Borrar todo — limpia todas las mediciones del mapa.
const medicion = {
  modo: null,        // null | "distancia" | "perpendicular" | "libre"
  drawer: null,      // instancia reutilizable de L.Draw.Polyline
  refLinea: null,    // capa elegida como referencia para la perpendicular
  refP1: null,       // extremo 1 del TRAMO exacto de referencia (no de toda la línea)
  refP2: null,       // extremo 2 del TRAMO exacto de referencia
  base: null,        // L.LatLng — punto base sobre la referencia
  preview: null,     // línea temporal mientras se mueve el ratón
  marcaBase: null,   // marcador temporal del punto base
  puntosLibres: [],  // vértices de la medición "desde coordenadas"
  trazoLibre: null,  // polilínea en curso de esa medición
};

function metrosPorGrado(lat) {
  return { lat: 111320, lon: 111320 * Math.cos(lat * Math.PI / 180) };
}

// Punto del segmento (p1→p2) más cercano a "click", en coordenadas reales.
function puntoMasCercanoEnLinea(p1, p2, click) {
  const m = metrosPorGrado(p1.lat);
  const dx = (p2.lng - p1.lng) * m.lon, dy = (p2.lat - p1.lat) * m.lat;
  const cx = (click.lng - p1.lng) * m.lon, cy = (click.lat - p1.lat) * m.lat;
  const largo2 = dx * dx + dy * dy;
  let t = largo2 === 0 ? 0 : (cx * dx + cy * dy) / largo2;
  t = Math.max(0, Math.min(1, t));
  return L.latLng(p1.lat + (dy * t) / m.lat, p1.lng + (dx * t) / m.lon);
}

// Distancia en PÍXELES de un clic a un segmento (tolerancia de clic igual
// a cualquier nivel de zoom). Devuelve también el propio segmento, para
// poder recordar exactamente cuál de los tramos de una línea se clicó.
function segmentoMasCercano(click, latlngs) {
  let mejor = null, distMin = Infinity;
  for (let i = 0; i < latlngs.length - 1; i++) {
    const [p1, p2] = [latlngs[i], latlngs[i + 1]];
    const c = state.mapa.latLngToLayerPoint(click);
    const a = state.mapa.latLngToLayerPoint(p1);
    const b = state.mapa.latLngToLayerPoint(p2);
    const dx = b.x - a.x, dy = b.y - a.y;
    const largo2 = dx * dx + dy * dy;
    let t = largo2 === 0 ? 0 : ((c.x - a.x) * dx + (c.y - a.y) * dy) / largo2;
    t = Math.max(0, Math.min(1, t));
    const d = Math.hypot(c.x - (a.x + dx * t), c.y - (a.y + dy * t));
    if (d < distMin) { distMin = d; mejor = { p1, p2 }; }
  }
  return { ...mejor, distMin };
}

// Punto perpendicular en vivo: proyecta el ratón sobre el eje perpendicular
// al tramo de referencia (refP1→refP2) que pasa por "base" — así el ángulo
// queda siempre fijo a 90° y solo cambia la distancia según el ratón, como
// una regla que se alarga.
function puntoPerpendicularVivo(p1, p2, base, mouse) {
  const m = metrosPorGrado(base.lat);
  const dx = (p2.lng - p1.lng) * m.lon, dy = (p2.lat - p1.lat) * m.lat;
  const largo = Math.sqrt(dx * dx + dy * dy) || 1;
  const ux = dx / largo, uy = dy / largo;
  const px = -uy, py = ux;

  const mx = (mouse.lng - base.lng) * m.lon, my = (mouse.lat - base.lat) * m.lat;
  const proy = mx * px + my * py;

  return L.latLng(base.lat + (py * proy) / m.lat, base.lng + (px * proy) / m.lon);
}

function guardarMedicion(linea, latlngs) {
  let total = 0;
  for (let i = 0; i < latlngs.length - 1; i++) total += state.mapa.distance(latlngs[i], latlngs[i + 1]);
  linea.addTo(state.layerMediciones);
  linea.bindTooltip(`${total < 1000 ? total.toFixed(1) + " m" : (total / 1000).toFixed(2) + " km"}`,
    { permanent: true, direction: "center", className: "" });
}

function botonesMedicion(modoActivo) {
  document.getElementById("btn-medir-distancia").className     = `w-full ${modoActivo === "distancia" ? "btn-primary" : "btn-secondary"}`;
  document.getElementById("btn-medir-perpendicular").className = `w-full ${modoActivo === "perpendicular" ? "btn-primary" : "btn-secondary"}`;
  document.getElementById("btn-medir-coords").className        = `w-full ${modoActivo === "libre" ? "btn-primary" : "btn-secondary"}`;
  const enLinea = modoActivo === "distancia" || modoActivo === "libre";
  document.getElementById("fila-deshacer-finalizar").classList.toggle("hidden", !enLinea);
}

function activarModoMedicion(modo) {
  finalizarModoMedicion();
  medicion.modo = modo;
  botonesMedicion(modo);

  if (modo === "distancia") {
    document.getElementById("medir-status").textContent = "Haz clic en el mapa para ir midiendo. Doble clic para terminar cada línea.";
    medicion.drawer.enable();
  } else if (modo === "perpendicular") {
    document.getElementById("medir-status").textContent = "Haz clic sobre una línea ya medida, cerca del tramo que quieras usar como referencia.";
    state.mapa.on("click", clickSeleccionarReferencia);
  }
}

function finalizarModoMedicion() {
  if (medicion.drawer && medicion.drawer._enabled) medicion.drawer.disable();
  state.mapa.off("click", clickSeleccionarReferencia);
  state.mapa.off("click", clickFijarBase);
  state.mapa.off("click", clickFijarDestino);
  state.mapa.off("click", clickAgregarPuntoLibre);
  if (medicion.preview)   { state.mapa.removeLayer(medicion.preview);   medicion.preview   = null; }
  if (medicion.marcaBase) { state.mapa.removeLayer(medicion.marcaBase); medicion.marcaBase = null; }
  if (medicion.trazoLibre && medicion.puntosLibres.length < 2) {
    state.mapa.removeLayer(medicion.trazoLibre); // se cancela si no llegó a tener 2 puntos guardados
  }
  if (medicion.refLinea) medicion.refLinea.setStyle({ color: "#f97316", weight: 3 });
  medicion.refLinea = null; medicion.refP1 = null; medicion.refP2 = null; medicion.base = null;
  medicion.trazoLibre = null; medicion.puntosLibres = [];
  medicion.modo = null;
  botonesMedicion(null);
}

function clickSeleccionarReferencia(e) {
  let elegida = null, distMin = Infinity, tramo = null;
  state.layerMediciones.eachLayer((capa) => {
    const r = segmentoMasCercano(e.latlng, capa.getLatLngs());
    if (r.distMin < distMin) { distMin = r.distMin; elegida = capa; tramo = r; }
  });
  if (!elegida || distMin > 20) {
    showBanner("Haz clic más cerca de una línea medida.", "warning");
    return;
  }
  medicion.refLinea = elegida;
  medicion.refP1 = tramo.p1;
  medicion.refP2 = tramo.p2;
  elegida.setStyle({ color: "#eab308", weight: 4 });
  state.mapa.off("click", clickSeleccionarReferencia);
  state.mapa.on("click", clickFijarBase);
  document.getElementById("medir-status").textContent =
    "Tramo de referencia elegido. Haz clic para fijar el punto base sobre él.";
}

function clickFijarBase(e) {
  medicion.base = puntoMasCercanoEnLinea(medicion.refP1, medicion.refP2, e.latlng);
  medicion.marcaBase = L.circleMarker(medicion.base, { radius: 5, color: "#eab308", fillColor: "#eab308", fillOpacity: 1 }).addTo(state.mapa);
  state.mapa.off("click", clickFijarBase);
  state.mapa.on("click", clickFijarDestino);
  document.getElementById("medir-status").textContent =
    "Mueve el ratón y haz clic para fijar el otro extremo (siempre a 90°).";
}

function clickFijarDestino(e) {
  const destino = puntoPerpendicularVivo(medicion.refP1, medicion.refP2, medicion.base, e.latlng);
  const linea = L.polyline([medicion.base, destino], { color: "#eab308", weight: 3 });
  guardarMedicion(linea, [medicion.base, destino]);
  document.getElementById("medir-status").textContent = "✅ Perpendicular medida. Elige otra herramienta o sigue midiendo.";
  finalizarModoMedicion();
}

// ── Medir desde coordenadas ───────────────────────────────────────
function iniciarMedicionDesdeCoordenadas() {
  const lat = parseFloat(document.getElementById("latitud").value);
  const lon = parseFloat(document.getElementById("longitud").value);
  if (isNaN(lat) || isNaN(lon)) {
    showBanner("Escribe unas coordenadas válidas en la tarjeta Coordenadas.", "warning");
    return;
  }
  finalizarModoMedicion();
  medicion.modo = "libre";
  botonesMedicion("libre");
  medicion.puntosLibres = [L.latLng(lat, lon)];
  medicion.trazoLibre = L.polyline(medicion.puntosLibres, { color: "#22c55e", weight: 3 }).addTo(state.mapa);
  medicion.marcaBase = L.circleMarker([lat, lon], { radius: 5, color: "#22c55e", fillColor: "#22c55e", fillOpacity: 1 }).addTo(state.mapa);
  state.mapa.setView([lat, lon], state.mapa.getZoom());
  document.getElementById("medir-status").textContent =
    "Punto de partida fijado en tus coordenadas. Haz clic para añadir más puntos y pulsa \"✓ Finalizar\" cuando termines.";
  state.mapa.on("click", clickAgregarPuntoLibre);
}

function clickAgregarPuntoLibre(e) {
  medicion.puntosLibres.push(e.latlng);
  medicion.trazoLibre.setLatLngs(medicion.puntosLibres);
  document.getElementById("medir-status").textContent =
    `${medicion.puntosLibres.length} puntos añadidos. Sigue haciendo clic o pulsa "✓ Finalizar".`;
}

function finalizarMedicionLibre() {
  if (medicion.puntosLibres.length < 2) {
    showBanner("Añade al menos un punto más en el mapa antes de finalizar.", "warning");
    return;
  }
  guardarMedicion(medicion.trazoLibre, medicion.puntosLibres);
  document.getElementById("medir-status").textContent = "✅ Línea medida. Elige otra herramienta o sigue midiendo.";
  finalizarModoMedicion();
}

// Deshacer y Finalizar funcionan igual en "Distancia" (tira de la
// herramienta nativa de Leaflet.draw) y en "Desde coordenadas" (nuestra
// versión propia), para que el usuario tenga un único par de botones.
function deshacerMedicionActual() {
  if (medicion.modo === "distancia") {
    medicion.drawer.deleteLastVertex();
  } else if (medicion.modo === "libre") {
    if (medicion.puntosLibres.length <= 1) {
      showBanner("Ese es el punto de partida; no se puede deshacer más.", "warning");
      return;
    }
    medicion.puntosLibres.pop();
    medicion.trazoLibre.setLatLngs(medicion.puntosLibres);
    document.getElementById("medir-status").textContent =
      `${medicion.puntosLibres.length} puntos. Sigue haciendo clic o pulsa "✓ Finalizar".`;
  }
}

function finalizarMedicionActual() {
  if (medicion.modo === "distancia") {
    medicion.drawer.completeShape();
  } else if (medicion.modo === "libre") {
    finalizarMedicionLibre();
  }
}

function inicializarHerramientaMedicion() {
  medicion.drawer = new L.Draw.Polyline(state.mapa, {
    shapeOptions: { color: "#f97316", weight: 3 },
    showLength: true,
    metric: true,
    repeatMode: true,
  });

  state.mapa.on(L.Draw.Event.CREATED, (e) => {
    if (e.layerType === "polyline") guardarMedicion(e.layer, e.layer.getLatLngs());
  });

  state.mapa.on("mousemove", (e) => {
    if (medicion.modo === "perpendicular" && medicion.base) {
      const destino = puntoPerpendicularVivo(medicion.refP1, medicion.refP2, medicion.base, e.latlng);
      if (medicion.preview) state.mapa.removeLayer(medicion.preview);
      medicion.preview = L.polyline([medicion.base, destino], { color: "#eab308", weight: 2, dashArray: "5 5" }).addTo(state.mapa);
    } else if (medicion.modo === "libre" && medicion.trazoLibre) {
      medicion.trazoLibre.setLatLngs([...medicion.puntosLibres, e.latlng]);
    }
  });

  // Escape cancela la medición en curso, igual que en SIGPAC.
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && medicion.modo) finalizarModoMedicion();
  });

  document.getElementById("btn-medir-distancia").addEventListener("click", () => {
    medicion.modo === "distancia" ? finalizarModoMedicion() : activarModoMedicion("distancia");
  });

  document.getElementById("btn-medir-perpendicular").addEventListener("click", () => {
    if (medicion.modo === "perpendicular") { finalizarModoMedicion(); return; }
    if (state.layerMediciones.getLayers().length === 0) {
      showBanner("Primero mide una línea con 📏, para usarla como referencia.", "warning");
      return;
    }
    activarModoMedicion("perpendicular");
  });

  document.getElementById("btn-medir-coords").addEventListener("click", () => {
    medicion.modo === "libre" ? finalizarModoMedicion() : iniciarMedicionDesdeCoordenadas();
  });

  document.getElementById("btn-medir-deshacer").addEventListener("click", deshacerMedicionActual);
  document.getElementById("btn-medir-finalizar").addEventListener("click", finalizarMedicionActual);

  document.getElementById("btn-medir-borrar").addEventListener("click", () => {
    finalizarModoMedicion();
    state.layerMediciones.clearLayers();
    document.getElementById("medir-status").textContent = "Elige una herramienta para empezar a medir.";
  });
}

// ── ANALIZAR PARCELA IA ──────────────────────────────────────────
document.getElementById("btn-analizar-parcela").addEventListener("click", async () => {
  if (!state.ultimasParcelas) { showBanner("Dibuja primero un polígono.", "warning"); return; }
  showSpinner("Analizando parcela con Sentinel-2... puede tardar 20-30 segundos");
  try {
    const res = await apiFetch("/analizar-parcela", { method: "POST", body: JSON.stringify({ coords: state.ultimasParcelas }) });
    if (!res.ok) throw new Error(await res.text());
    const d = await res.json();

    state.layerIA.clearLayers();
    if (d.puntos_heatmap && d.puntos_heatmap.length > 0) {
      // La puntuación es relativa a la parcela: el color se escala con su
      // máximo, para que la zona más favorable salga en rojo y el mapa se vea
      // aunque las puntuaciones sean bajas (antes, con max 1.0, salía casi transparente).
      const maxCalor = Math.max(...d.puntos_heatmap.map(p => p[2]), 0.01);
      state.layerIA.addLayer(L.heatLayer(d.puntos_heatmap, {
        radius: 35, blur: 25, maxZoom: 18, max: maxCalor, minOpacity: 0.35,
        gradient: { 0.0: "#0000ff", 0.25: "#00ffff", 0.5: "#00ff00", 0.75: "#ffff00", 1.0: "#ff0000" },
      }));
    }
    L.marker([d.pozo_lat, d.pozo_lon], {
      icon: L.divIcon({ className: "", html: `<div style="font-size:28px;line-height:1;color:#111827;text-shadow:0 0 3px #fff,0 0 3px #fff;">★</div>`, iconAnchor: [14, 14] }),
    }).bindTooltip(`⚫ Punto óptimo BALSA (agua superficial) — ${d.pozo_score}%`).addTo(state.layerIA);
    state.puntoBalsa = { lat: d.pozo_lat, lon: d.pozo_lon, score: d.pozo_score };

    document.getElementById("leyenda-ia").classList.remove("hidden");
    document.getElementById("btn-limpiar").classList.remove("hidden");

    document.getElementById("metricas-ia").innerHTML =
      metricCard("Puntuación óptima", `${d.pozo_score}%`) + metricCard("Calidad datos", d.calidad) +
      (d.ndvi      !== null ? metricCard("NDVI zona óptima", d.ndvi.toFixed(3))       : "") +
      (d.humedad   !== null ? metricCard("Humedad ERA5",     d.humedad.toFixed(4))    : "") +
      (d.elevacion !== null ? metricCard("Elevación",        `${Math.round(d.elevacion)} m`) : "");

    document.getElementById("calidad-info").textContent = `📡 ${d.calidad_txt}`;
    document.getElementById("coord-optima").textContent = "";   // las coordenadas van en el recuadro de resultado

    const razones = [];
    if (d.ndvi     && d.ndvi     > 0.4) razones.push("vegetación densa que indica humedad profunda");
    if (d.humedad  && d.humedad  > 0.3) razones.push("alta retención de humedad en el suelo");
    if (d.elevacion && d.elevacion < 300) razones.push("zona baja donde el agua tiende a acumularse");
    razones.push("convergencia topográfica favorable");
    document.getElementById("explicacion-ia").textContent = "💡 El índice ponderado de agua superficial señala esta zona porque presenta " + razones.join(", ") + ".";

    document.getElementById("resumen-puntos").innerHTML = `
      <div class="mt-2 p-3 rounded-lg" style="background:rgba(0,0,0,0.3);border:1px solid rgba(255,255,255,0.1)">
        <p class="text-xs font-semibold text-white/80 mb-1">🌡️ Cómo leer el mapa de calor:</p>
        <div class="flex items-center gap-2 mb-1">
          <div style="width:60px;height:8px;border-radius:4px;background:linear-gradient(to right,#00f,#0ff,#0f0,#ff0,#f00)"></div>
          <span class="text-xs text-white/60">Azul = puntuación baja · Rojo = puntuación alta</span>
        </div>
        <p class="text-xs text-white/50">⚫ Estrella negra = punto óptimo para BALSA (agua superficial).</p>
        <p class="text-xs text-white/50 mt-1">📡 ${d.puntos_heatmap ? d.puntos_heatmap.length : 0} puntos analizados con Sentinel-2.</p>
      </div>`;

    // ── Punto óptimo para POZO PROFUNDO (agua subterránea) ───────────
    // Llamada independiente: si falla, no afecta a nada de lo ya mostrado
    // arriba (punto óptimo de balsa, mapa de calor, geología general).
    let puntoPozo = null;
    try {
      const resPozo = await apiFetch("/punto-pozo-profundo", { method: "POST", body: JSON.stringify({ coords: state.ultimasParcelas }) });
      if (resPozo.ok) {
        const dp = await resPozo.json();
        if (dp.disponible) {
          puntoPozo = { lat: dp.pozo_profundo_lat, lon: dp.pozo_profundo_lon };
          L.marker([dp.pozo_profundo_lat, dp.pozo_profundo_lon], {
            icon: L.divIcon({ className: "", html: `<div style="font-size:28px;line-height:1;color:#d4af37;text-shadow:0 0 3px #000,0 0 3px #000;">★</div>`, iconAnchor: [14, 14] }),
          }).bindTooltip(`🌟 Punto óptimo POZO PROFUNDO (agua subterránea) — ${dp.pozo_profundo_score}%`).addTo(state.layerIA);
          dibujarAlternativasPozo(dp, state.layerIA);

          guardarPuntoPozo(dp);
        }
      }
    } catch (errPozo) {
      console.warn("No se pudo calcular el punto óptimo de pozo profundo:", errPozo);
    }

    if (d.geologia) {
      const g = d.geologia;
      const geoMsg = { buena: alertHTML("success","✅ Geología <strong>FAVORABLE</strong>."), mala: alertHTML("danger","❌ Geología <strong>DESFAVORABLE</strong>."), media: alertHTML("warning","⚠️ Geología <strong>MODERADA</strong>.") };
      document.getElementById("geologia-parcela").innerHTML = `
        <h3 class="card-title mt-1">🪨 Geología del subsuelo</h3>
        <div class="mt-2 text-sm text-white/80">
          <p><strong class="text-white">Unidad geológica:</strong> ${g.nombre_unidad}</p>
          <p><strong class="text-white">Tipo de roca:</strong> ${g.roca}</p>
          <p class="text-white/60 text-xs mt-1">${g.descripcion}</p>
        </div>${geoMsg[g.estado] || ""}`;
      document.getElementById("geologia-parcela").classList.remove("hidden");
    }

    document.getElementById("card-ia").classList.remove("hidden");
    pintarResultadoPuntos();
    (document.getElementById("card-resultado-puntos") || document.getElementById("card-ia")).scrollIntoView({ behavior: "smooth" });
    // Profundidad del agua subterránea con su intervalo de error, en la
    // estrella dorada si existe y, si no, en el centro de la parcela
    consultarProfundidad(state.ultimasParcelas, puntoPozo);

  } catch (err) { showBanner(`❌ Error análisis IA: ${err.message}`); }
  finally { hideSpinner(); }
});

// ── VER VAGUADAS ─────────────────────────────────────────────────
document.getElementById("btn-vaguadas").addEventListener("click", async () => {
  if (!state.ultimasParcelas) { showBanner("Dibuja primero un polígono.", "warning"); return; }
  showSpinner("Calculando flujo de agua... ~30 segundos");
  try {
    const res = await apiFetch("/vaguadas", { method: "POST", body: JSON.stringify({ coords: state.ultimasParcelas }) });
    if (!res.ok) throw new Error(await res.text());
    const d = await res.json();
    state.layerVaguadas.clearLayers();

    // Líneas de drenaje — escorrentía (flujo difuso) en cian, vaguada (cauce
    // concentrado, acumulación >= VAGUADA_THR) en índigo, para distinguirlas
    // a simple vista. Cada una muestra su caudal estimado (Método Racional).
    const COLOR_ESCORRENTIA = "#38bdf8", COLOR_VAGUADA = "#4f46e5";
    const lineas = { type: "FeatureCollection", features: d.geojson.features.filter(f => f.properties.tipo === "linea_drenaje") };
    const geojsonLayer = L.geoJSON(lineas, {
      style: f => ({
        color: f.properties.categoria === "vaguada" ? COLOR_VAGUADA : COLOR_ESCORRENTIA,
        weight: Math.max(1, Math.round((f.properties.intensidad||0.3)*5)),
        opacity: 0.85,
      }),
      onEachFeature: (f, l) => {
        const esVaguada = f.properties.categoria === "vaguada";
        l.bindTooltip(
          `${esVaguada ? "🟣 Vaguada" : "🔵 Escorrentía"} — ${f.properties.acumulacion} celdas ` +
          `(${f.properties.area_ha} ha) — caudal est.: ${f.properties.caudal_ls} L/s`
        );
      },
    }).addTo(state.layerVaguadas);

    // Flecha de sentido del flujo — cada línea de drenaje va de la celda
    // con más acumulación (origen) a la celda vecina donde vierte (backend
    // ya las construye en ese orden), así que basta con orientar la flecha
    // de coordenadas[0] a coordenadas[1] de cada segmento. Usa el mismo
    // color (aclarado) que la línea a la que pertenece.
    geojsonLayer.eachLayer((capa) => {
      const [[lon1, lat1], [lon2, lat2]] = capa.feature.geometry.coordinates;
      const esVaguada = capa.feature.properties.categoria === "vaguada";
      const colorFlecha = esVaguada ? "#c7d2fe" : "#bae6fd";
      const p1 = L.latLng(lat1, lon1), p2 = L.latLng(lat2, lon2);
      const medio = L.latLng((lat1 + lat2) / 2, (lon1 + lon2) / 2);
      const a = state.mapa.latLngToLayerPoint(p1), b = state.mapa.latLngToLayerPoint(p2);
      const angulo = Math.atan2(b.y - a.y, b.x - a.x) * 180 / Math.PI;
      // Triángulo CSS centrado dentro de un contenedor holgado (28x28): al
      // rotar en un ángulo cualquiera, la forma nunca llega a tocar el
      // borde del icono, así que no puede quedar recortada.
      L.marker(medio, {
        interactive: false,
        icon: L.divIcon({
          className: "",
          html: `<div style="width:28px;height:28px;display:flex;align-items:center;justify-content:center;">
                   <div style="width:0;height:0;
                        border-top:6px solid transparent;
                        border-bottom:6px solid transparent;
                        border-left:11px solid ${colorFlecha};
                        transform:rotate(${angulo}deg);
                        filter:drop-shadow(0 0 1.5px #000) drop-shadow(0 0 1.5px #000);"></div>
                 </div>`,
          iconSize: [28, 28],
          iconAnchor: [14, 14],
        }),
      }).addTo(state.layerVaguadas);
    });

    // Puntos de acumulación — se numeran UNA sola vez (mayor a menor
    // acumulación) y se usa el MISMO número en el mapa y en la lista de
    // abajo, para que se vea claramente qué punto del mapa es cada fila.
    const puntos = d.geojson.features
      .filter(f => f.properties.tipo === "punto_acumulacion")
      .sort((a, b) => b.properties.acumulacion - a.properties.acumulacion);
    state.puntosBalsaD8 = puntos.slice(0, 3).map(f => ({
      lat: f.geometry.coordinates[1], lon: f.geometry.coordinates[0], acum: f.properties.acumulacion }));

    puntos.forEach((f, i) => {
      const [lon, lat] = f.geometry.coordinates;
      const numero  = i + 1;
      const esMejor = i === 0;
      const marker = L.marker([lat, lon], {
        icon: L.divIcon({
          className: "",
          html: `<div style="width:26px;height:26px;border-radius:50%;
                 background:${esMejor ? "#f59e0b" : "#0ea5e9"};
                 border:2px solid ${esMejor ? "#fde68a" : "#0369a1"};
                 color:#fff;font-weight:700;font-size:12px;
                 display:flex;align-items:center;justify-content:center;
                 box-shadow:0 1px 4px rgba(0,0,0,0.5);">${esMejor ? "⭐" : numero}</div>`,
          iconAnchor: [13, 13],
        }),
      })
        .bindTooltip(`Punto ${numero}${esMejor ? " — mejor zona" : ""} — ${f.properties.etiqueta || ""}`)
        .addTo(state.layerVaguadas);

      // Clic en el marcador → resalta y hace scroll a su fila en la lista.
      marker.on("click", () => {
        const card = document.getElementById(`punto-vaguada-${numero}`);
        if (!card) return;
        card.scrollIntoView({ behavior: "smooth", block: "center" });
        card.style.outline = "2px solid #fbbf24";
        setTimeout(() => { card.style.outline = ""; }, 1500);
      });
    });

    if (state.layerVaguadas.getLayers().length) state.mapa.fitBounds(geojsonLayer.getBounds(),{padding:[20,20]});
    document.getElementById("leyenda-vaguadas").classList.remove("hidden");
    document.getElementById("btn-limpiar").classList.remove("hidden");
    document.getElementById("metricas-vaguadas").innerHTML =
      metricCard("Líneas drenaje", d.n_lineas) +
      metricCard("Vaguadas", d.n_vaguadas) +
      metricCard("Puntos acumulación", d.n_puntos) +
      metricCard("Acum. máxima", `${d.acum_maxima} celdas`) +
      metricCard("Caudal máx. est.", `${d.caudal_maximo_ls} L/s`);
    document.getElementById("info-vaguadas").innerHTML =
      `<p class="text-white/60 text-xs">📡 ${d.resolucion}</p>` +
      `<p class="mt-2"><span style="color:#38bdf8;">■</span> Escorrentía (flujo difuso) &nbsp; ` +
      `<span style="color:#4f46e5;">■</span> Vaguada (cauce concentrado). ` +
      `El caudal de cada tramo es una estimación por el Método Racional (Q=C·I·A/360), ` +
      `con coeficiente de escorrentía y lluvia de referencia asumidos — orientativo, no una medición real.</p>` +
      `<p class="mt-2">Cada círculo numerado del mapa corresponde al mismo número en la lista de abajo — haz clic en uno para localizar el otro.</p>`;

    // Estrella dorada + chequeo de acuífero (geología real + pozos reales
    // del IGME cercanos) — mismo cálculo que "Analizar parcela con IA",
    // añadido aquí para no depender de pulsar ese otro botón.
    if (d.pozo_acuifero && d.pozo_acuifero.disponible) {
      const dp = d.pozo_acuifero;
      L.marker([dp.pozo_profundo_lat, dp.pozo_profundo_lon], {
        icon: L.divIcon({ className: "", html: `<div style="font-size:28px;line-height:1;color:#d4af37;text-shadow:0 0 3px #000,0 0 3px #000;">★</div>`, iconAnchor: [14, 14] }),
      }).bindTooltip(`🌟 Punto óptimo POZO (agua subterránea) — ${dp.pozo_profundo_score}%`).addTo(state.layerVaguadas);
      dibujarAlternativasPozo(dp, state.layerVaguadas);

      guardarPuntoPozo(dp);
      pintarResultadoPuntos();
    } else if (d.pozo_acuifero) {
      document.getElementById("info-vaguadas").innerHTML += `
        <p class="mt-2 text-xs text-white/40">🌟 Chequeo de acuífero no disponible ahora mismo (${d.pozo_acuifero.mensaje || "sin datos suficientes"}).</p>`;
    }

    // Lista de puntos — mismo orden y numeración que en el mapa, con id
    // para poder hacer scroll hasta aquí al clicar un marcador.
    document.getElementById("puntos-vaguadas").innerHTML = puntos.map((f, i) => {
      const [lon, lat] = f.geometry.coordinates;
      const numero = i + 1;
      return `<div id="punto-vaguada-${numero}" class="punto-card" style="transition:outline 0.2s;"><strong>${i===0?"⭐":numero+"."} Punto ${numero}</strong> — ${f.properties.etiqueta}<br/><span class="text-white/50 text-xs">Lat:${lat.toFixed(5)}, Lon:${lon.toFixed(5)} · ${f.properties.acumulacion} celdas</span></div>`;
    }).join("");

    // Clic en una fila de la lista → centra el mapa en su punto.
    puntos.forEach((f, i) => {
      const numero = i + 1;
      const [lon, lat] = f.geometry.coordinates;
      document.getElementById(`punto-vaguada-${numero}`)?.addEventListener("click", () => {
        state.mapa.setView([lat, lon], Math.max(state.mapa.getZoom(), 17));
      });
    });

    pintarResultadoPuntos();
    document.getElementById("card-vaguadas").classList.remove("hidden");
    document.getElementById("card-vaguadas").scrollIntoView({behavior:"smooth"});
  } catch (err) { showBanner(`❌ Error vaguadas: ${err.message}`); }
  finally { hideSpinner(); }
});

// ── SERIE TEMPORAL NDVI ──────────────────────────────────────────
document.getElementById("btn-ndvi-serie").addEventListener("click", async () => {
  if (!state.ultimasParcelas) { showBanner("Dibuja primero un polígono.","warning"); return; }
  showSpinner("Calculando NDVI 2021-2024... puede tardar 30-60 segundos");
  try {
    const d = await esperarTrabajo("/trabajos/ndvi-serie", { coords: state.ultimasParcelas });
    if (!d.serie?.length) { showBanner("Sin imágenes satelitales para esta zona.","warning"); return; }
    if (state.graficaNDVI) { state.graficaNDVI.destroy(); state.graficaNDVI = null; }
    const ctx = document.getElementById("grafica-ndvi").getContext("2d");
    state.graficaNDVI = new Chart(ctx, {
      type:"line",
      data:{ labels:d.serie.map(p=>p.fecha), datasets:[{ label:"NDVI medio parcela", data:d.serie.map(p=>p.ndvi), borderColor:"#4ade80", backgroundColor:"rgba(74,222,128,0.15)", borderWidth:2, pointRadius:3, pointBackgroundColor:d.serie.map(p=>[6,7,8].includes(p.mes)?"#fb923c":"#4ade80"), fill:true, tension:0.3 }] },
      options:{ responsive:true, maintainAspectRatio:false, plugins:{ legend:{labels:{color:"#fff",font:{size:11}}}, tooltip:{callbacks:{label:ctx=>` NDVI: ${ctx.parsed.y}`, afterLabel:ctx=>[6,7,8].includes(d.serie[ctx.dataIndex].mes)?"☀️ Verano":""}} }, scales:{ x:{ticks:{color:"#9ca3af",maxRotation:45,font:{size:9}},grid:{color:"rgba(255,255,255,0.06)"}}, y:{min:0,max:1,ticks:{color:"#9ca3af"},grid:{color:"rgba(255,255,255,0.06)"}} } }
    });
    document.getElementById("ndvi-firma").innerHTML = d.tiene_firma
      ? alertHTML("success",`💧 <strong>FIRMA DE ACUÍFERO DETECTADA</strong> — ${d.meses_verano_alto} meses de verano con NDVI alto.`)
      : alertHTML("info","ℹ️ No se detecta firma clara de acuífero en verano.");
    document.getElementById("ndvi-info").textContent = `📡 ${d.n_imagenes} imágenes Sentinel-2 · Puntos naranjos = meses de verano`;
    document.getElementById("card-ndvi").classList.remove("hidden");
    document.getElementById("btn-limpiar").classList.remove("hidden");
    document.getElementById("card-ndvi").scrollIntoView({behavior:"smooth"});
  } catch (err) { showBanner(`❌ Error NDVI: ${err.message}`); }
  finally { hideSpinner(); }
});

// ── ANOMALÍAS TÉRMICAS ───────────────────────────────────────────
document.getElementById("btn-thermal").addEventListener("click", async () => {
  if (!state.ultimasParcelas) { showBanner("Dibuja primero un polígono.","warning"); return; }
  showSpinner("Analizando temperatura con Landsat... 20-30 segundos");
  try {
    const res = await apiFetch("/thermal", { method:"POST", body: JSON.stringify({coords:state.ultimasParcelas}) });
    if (!res.ok) throw new Error(await res.text());
    const d = await res.json();
    state.layerThermal.clearLayers();
    if (d.puntos_frios?.length) {
      const bounds = [];
      d.puntos_frios.forEach(pt => {
        L.circleMarker([pt.lat,pt.lon],{radius:8,color:"#06b6d4",fillColor:"#22d3ee",fillOpacity:0.85,weight:1.5}).bindTooltip(`🌡️ ${pt.temp_c}°C — posible acuífero`).addTo(state.layerThermal);
        bounds.push([pt.lat,pt.lon]);
      });
      if (bounds.length) state.mapa.fitBounds(L.latLngBounds(bounds),{padding:[30,30]});
    }
    document.getElementById("metricas-thermal").innerHTML = metricCard("Temp. media",d.temp_media!==null?`${d.temp_media}°C`:"N/D")+metricCard("Temp. mínima",d.temp_min!==null?`${d.temp_min}°C`:"N/D")+metricCard("Zonas frescas",`${d.puntos_frios.length} puntos`);
    document.getElementById("thermal-info").innerHTML = (d.puntos_frios.length>0 ? alertHTML("info",`🌡️ <strong>${d.puntos_frios.length} zonas frescas</strong> detectadas. Las zonas más frías en agosto suelen indicar humedad subterránea.`) : alertHTML("warning","⚠️ No se detectaron anomalías térmicas.")) + `<p class="text-white/40 text-xs mt-2">📡 ${d.n_imagenes} imágenes Landsat (verano 2022-2024)</p>`;
    document.getElementById("card-thermal").classList.remove("hidden");
    document.getElementById("btn-limpiar").classList.remove("hidden");
    document.getElementById("card-thermal").scrollIntoView({behavior:"smooth"});
  } catch (err) { showBanner(`❌ Error térmico: ${err.message}`); }
  finally { hideSpinner(); }
});

// ── MODELOS DE IA (solo admin ve el panel de entrenamiento) ──
if (SESSION.role === "admin") {
  document.getElementById("panel-modelo-admin")?.classList.remove("hidden");

  document.getElementById("input-csv")?.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    document.getElementById("modelo-estado").innerHTML = alertHTML("info","⏳ Entrenando modelo...");
    showSpinner("Entrenando modelo de IA... 1-2 minutos");
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await apiFetchForm("/entrenar", formData);
      if (!res.ok) throw new Error(await res.text());
      const d = await res.json();
      const m = d.metricas.metricas_cv;
      const interp = d.metricas.interpretacion;
      document.getElementById("modelo-estado").innerHTML = alertHTML("success",
        `✅ Entrenado con ${d.metricas.n_pozos_entrenamiento} pozos — Calidad: <strong>${interp.calidad_modelo}</strong> · R²: <strong>${Number(m.r2_medio).toFixed(2)}</strong> · RMSE: <strong>${m.rmse_medio} m</strong>`);
      mostrarMetricasCompletas(d.metricas);
    } catch (err) {
      document.getElementById("modelo-estado").innerHTML = alertHTML("danger",`❌ Error: ${err.message}`);
    } finally { hideSpinner(); e.target.value = ""; }
  });

  document.getElementById("btn-ver-metricas")?.addEventListener("click", async () => {
    try {
      const res = await apiFetch("/modelo-info");
      const d   = await res.json();
      if (!d.disponible && !d.unet) { document.getElementById("modelo-estado").innerHTML = alertHTML("warning","⚠️ Modelo no entrenado."); return; }
      mostrarMetricasCompletas(d.disponible ? d.metricas : null, d.unet);
    } catch (err) { showBanner(`❌ ${err.message}`); }
  });
}

function mostrarMetricasCompletas(metricas, unet) {
  const div = document.getElementById("metricas-modelo");
  if (!div) return;
  let html = "";
  if (metricas) {
    const m = metricas.metricas_cv, interp = metricas.interpretacion, imp = metricas.importancia_variables, iv = metricas.intervalo;
    html += `
      <p class="text-xs font-semibold text-white/80 mb-2">🧠 Profundidad del agua (Gradient Boosting)</p>
      <div class="grid grid-cols-2 gap-2">
        ${metricCard("Error medio (MAE)", m.mae_m !== undefined ? m.mae_m + " m" : m.rmse_medio + " m (RMSE)")}
        ${metricCard("R² (validación espacial)", Number(m.r2_global ?? m.r2_medio).toFixed(2))}
        ${metricCard("Cobertura del intervalo", iv ? `${Math.round(iv.cobertura * 100)} % (objetivo ${Math.round(iv.confianza * 100)} %)` : "N/D")}
        ${metricCard("Mejora frente a IDW", interp.mejora_frente_idw_pct !== undefined ? interp.mejora_frente_idw_pct + " %" : "N/D")}
      </div>
      <div class="mt-2 text-xs text-white/70">${interp.descripcion}</div>
      <div class="mt-1 text-xs text-white/70">${interp.r2_texto}</div>
      ${interp.intervalo_texto ? `<div class="mt-1 text-xs text-white/70">${interp.intervalo_texto}</div>` : ""}
      ${metricas.comparacion ? `<div class="mt-2 text-xs text-white/60">${Object.entries(metricas.comparacion).map(([n, v]) => `${n}: MAE ${v.mae_m} m`).join(" · ")}</div>` : ""}
      <div class="mt-3"><p class="text-xs font-semibold text-white/80 mb-2">Importancia de variables:</p>
        ${Object.entries(imp).map(([n,v])=>`<div class="mb-1"><div class="flex justify-between text-xs text-white/60"><span>${n}</span><span>${Math.round(v*100)}%</span></div><div style="background:rgba(96,165,250,0.5);height:6px;border-radius:3px;width:${Math.min(v*100,100)}%;margin-top:2px;"></div></div>`).join("")}
      </div>
      <div class="mt-2 text-xs text-white/40">Entrenado con ${metricas.n_pozos_entrenamiento} pozos · ${m.tipo_validacion || ""}</div>`;
  }
  if (unet && unet.test) {
    html += `
      <p class="text-xs font-semibold text-white/80 mt-4 mb-2">💧 Agua superficial (U-Net)</p>
      <div class="grid grid-cols-1 gap-2">
        ${Object.entries(unet.test).map(([n, v]) => metricCard(n, `IoU ${fmtPct(v.iou)} · F1 ${fmtPct(v.f1_dice)} · precisión ${fmtPct(v.precision)} · sensibilidad ${fmtPct(v.recall)}`)).join("")}
      </div>
      <div class="mt-2 text-xs text-white/40">${unet.n_parches.entrenamiento + unet.n_parches.validacion + unet.n_parches.test} parches · ${unet.separacion} · ${Number(unet.parametros_entrenables).toLocaleString("es-ES")} parámetros · mejor época ${unet.mejor_epoca}</div>`;
  }
  div.innerHTML = html || alertHTML("warning", "⚠️ Ningún modelo entrenado.");
  div.classList.remove("hidden");
}

// ── PROFUNDIDAD DEL AGUA SUBTERRÁNEA (Gradient Boosting + intervalo) ──
function fmtPct(v) {
  return (v === null || v === undefined) ? "N/D" : `${Math.round(v * 100)} %`;
}

function htmlProfundidad(d) {
  const colores = { verde: "success", amarillo: "warning", rojo: "danger" };
  const min = d.intervalo_min_m, max = d.intervalo_max_m, est = d.profundidad_estimada_m;
  const escala = Math.max(max * 1.1, 10);
  const pct = v => Math.min(100, (v / escala) * 100);
  const categoria = (d.veredicto || "").split(":")[0];
  return alertHTML(colores[d.color] || "info",
      `🧠 Profundidad estimada del agua: <strong>~${Math.round(est)} m</strong> (${categoria})<br>` +
      `<span class="text-xs">Entre <strong>${Math.round(min)} y ${Math.round(max)} m</strong> con un ${d.confianza_pct} % de confianza</span>`) + `
    <div class="mt-2 p-2 rounded-lg" style="background:rgba(0,0,0,0.3)">
      <div style="position:relative;height:14px;background:rgba(255,255,255,0.1);border-radius:7px">
        <div style="position:absolute;left:${pct(min)}%;width:${Math.max(pct(max) - pct(min), 1)}%;height:100%;background:rgba(96,165,250,0.6);border-radius:7px"></div>
        <div style="position:absolute;left:calc(${pct(est)}% - 2px);width:4px;height:100%;background:#fff;border-radius:2px"></div>
      </div>
      <div class="flex justify-between text-xs text-white/50 mt-1"><span>0 m</span><span>${Math.round(escala)} m</span></div>
      <p class="text-xs text-white/70 mt-2">${d.descripcion}</p>
      <p class="text-xs text-white/60 mt-1">📍 Pozo del IGME más cercano a ${d.distancia_pozo_cercano_km} km · media ponderada de los 5 más cercanos: ${d.profundidad_idw_vecinos_m} m (puntos amarillos en el mapa)</p>
      ${(d.avisos || []).map(a => `<p class="text-xs mt-1" style="color:#fde047">⚠️ ${a}</p>`).join("")}
    </div>
    <p class="text-xs text-white/40 mt-1">Gradient Boosting entrenado con pozos reales del IGME · intervalo conformal validado en zonas no vistas</p>`;
}

async function consultarProfundidad(coords, punto) {
  const cuerpo = punto ? { latitud: punto.lat, longitud: punto.lon } : { coords };
  try {
    const res = await apiFetch("/profundidad", { method: "POST", body: JSON.stringify(cuerpo) });
    if (!res?.ok) return;
    const d = await res.json();
    const div = document.getElementById("resultado-modelo");
    if (!d.disponible) {
      div.innerHTML = alertHTML("warning", `⚠️ ${d.mensaje || "Modelo de profundidad no disponible."}`);
    } else {
      div.innerHTML = `<p class="text-xs text-white/60 mb-1">${punto ? "Estimación en el punto óptimo para pozo (estrella dorada)" : "Estimación en el centro de la parcela"}</p>` + htmlProfundidad(d);
      (d.pozos_vecinos || []).forEach(p => {
        L.circleMarker([p.lat, p.lon], { radius: 6, color: "#a16207", fillColor: "#fde047", fillOpacity: 0.9, weight: 1 })
          .bindTooltip(`Pozo IGME: ${p.profundidad_m} m de profundidad (a ${p.distancia_km} km)`).addTo(state.layerIA);
      });
    }
    document.getElementById("card-prediccion-modelo").classList.remove("hidden");
  } catch (err) { console.warn("No se pudo estimar la profundidad:", err); }
}

async function anadirProfundidadModeloPunto(lat, lon) {
  try {
    const res = await apiFetch("/profundidad", { method: "POST", body: JSON.stringify({ latitud: lat, longitud: lon }) });
    if (!res?.ok) return;
    const d = await res.json();
    if (!d.disponible) return;
    document.getElementById("metricas-punto").innerHTML +=
      metricCard("🧠 Profundidad (modelo IA)", `${Math.round(d.profundidad_estimada_m)} m (${Math.round(d.intervalo_min_m)}–${Math.round(d.intervalo_max_m)} m)`);
    document.getElementById("detalle-profundidad").innerHTML +=
      `<p class="mt-2"><strong>Modelo IA (Gradient Boosting):</strong> ~${Math.round(d.profundidad_estimada_m)} m, ` +
      `entre ${Math.round(d.intervalo_min_m)} y ${Math.round(d.intervalo_max_m)} m con un ${d.confianza_pct} % de confianza.</p>`;
  } catch (err) { console.warn("No se pudo estimar la profundidad del punto:", err); }
}

// ── RECUADRO DE RESULTADO: punto para balsa + punto para pozo ────
// Reúne en un solo sitio los dos puntos recomendados (estrella negra y
// estrella dorada) para que no se repitan en varias tarjetas.
const GEOLOGIA_TXT = { buena: "favorable", media: "moderada", mala: "desfavorable" };

function guardarPuntoPozo(dp) {
  state.puntoPozo = {
    lat: dp.pozo_profundo_lat, lon: dp.pozo_profundo_lon, score: dp.pozo_profundo_score,
    geologia: dp.geologia ? `${dp.geologia.nombre_unidad} (${GEOLOGIA_TXT[dp.geologia.estado] || dp.geologia.estado})` : null,
    igme: dp.distancia_pozo_real_m !== null && dp.distancia_pozo_real_m !== undefined
      ? `Pozo real del IGME más cercano a ${Math.round(dp.distancia_pozo_real_m)} m`
      : "Sin pozos reales del IGME cerca",
    otros: (dp.candidatos_pozo || []).filter(c => !c.principal),
  };
}

// Alternativas para pozo: círculos dorados numerados (la estrella dorada es la nº 1)
function dibujarAlternativasPozo(dp, capa) {
  (dp.candidatos_pozo || []).filter(c => !c.principal).forEach((c, i) => {
    const n = i + 2;
    L.marker([c.lat, c.lon], {
      icon: L.divIcon({ className: "", iconSize: [18, 18], iconAnchor: [9, 9],
        html: `<div style="width:18px;height:18px;border-radius:50%;background:#d4af37;border:1px solid #111;color:#111;font:bold 11px sans-serif;display:flex;align-items:center;justify-content:center">${n}</div>` }),
    }).bindTooltip(`Alternativa ${n} para POZO (agua subterránea) — ${c.score}%`).addTo(capa);
  });
}

function crearUIResultadoPuntos() {
  if (document.getElementById("card-resultado-puntos")) return;
  const ref = document.getElementById("card-prediccion-modelo") || document.getElementById("card-ia");
  if (!ref) return;
  const card = document.createElement("div");
  card.id = "card-resultado-puntos";
  card.className = (document.getElementById("card-ia") || ref).className;   // mismo estilo que las demás tarjetas
  card.classList.add("hidden");
  card.style.border = "2px solid #fbbf24";
  card.style.boxShadow = "0 0 18px rgba(251,191,36,0.35)";
  card.innerHTML = `
    <h3 class="card-title">📍 Resultado: dónde buscar agua en la parcela</h3>
    <div id="resultado-puntos-cuerpo"></div>`;
  ref.insertAdjacentElement("beforebegin", card);
}

function pintarResultadoPuntos() {
  try { pintarResultadoPuntosSeguro(); }
  catch (err) { console.warn("No se pudo pintar el recuadro de resultado:", err); }
}

function pintarResultadoPuntosSeguro() {
  crearUIResultadoPuntos();
  const card = document.getElementById("card-resultado-puntos");
  const cuerpo = document.getElementById("resultado-puntos-cuerpo");
  if (!card || !cuerpo) return;
  const b = state.puntoBalsa, p = state.puntoPozo, d8 = state.puntosBalsaD8 || [];
  if (!b && !p && !d8.length) { card.classList.add("hidden"); return; }
  const bloque = (fondo, borde, titulo, contenido) => `
    <div class="mt-2 p-3 rounded-lg" style="background:${fondo};border:1px solid ${borde}">
      <p class="text-sm font-semibold text-white">${titulo}</p>${contenido}
    </div>`;
  const coord = (lat, lon) => `${lat.toFixed(6)}, ${lon.toFixed(6)}`;
  const lista = (items, inicio = 1) =>
    `<ol start="${inicio}" class="text-xs text-white/75 mt-1" style="padding-left:1.2rem;list-style:decimal">${items.join("")}</ol>`;

  let balsa = b
    ? `<p class="text-xs text-white/85 mt-1">⚫ <strong>Principal</strong> (estrella negra): 📍 ${coord(b.lat, b.lon)} · Puntuación: <strong>${b.score}%</strong></p>
       <p class="text-xs text-white/55">La zona de la parcela más favorable para acumular agua en superficie.</p>`
    : `<p class="text-xs text-white/55 mt-1">Pulsa «Analizar parcela con IA» para calcular el punto principal.</p>`;
  balsa += d8.length
    ? `<p class="text-xs text-white/70 mt-2">Otros puntos donde se junta la escorrentía (círculos numerados del D8):</p>` +
      lista(d8.map(q => `<li>📍 ${coord(q.lat, q.lon)} · acumulación de ${Math.round(q.acum)} celdas</li>`))
    : `<p class="text-xs text-white/45 mt-2">Pulsa «Ver flujo de agua y vaguadas» para ver más puntos para balsa.</p>`;

  const pozo = p
    ? `<p class="text-xs text-white/85 mt-1">🌟 <strong>Principal</strong> (estrella dorada): 📍 ${coord(p.lat, p.lon)} · Puntuación: <strong>${p.score}%</strong></p>
       ${p.geologia ? `<p class="text-xs text-white/65 mt-1">Geología del punto: ${p.geologia}</p>` : ""}
       <p class="text-xs text-white/65">${p.igme}</p>
       <p class="text-xs text-white/55">Vegetación verde y suelo fresco en verano, cerca de pozos reales.</p>` +
      ((p.otros || []).length
        ? `<p class="text-xs text-white/70 mt-2">Alternativas para pozo (círculos dorados numerados):</p>` +
          lista(p.otros.map(c => `<li>📍 ${coord(c.lat, c.lon)} · Puntuación: ${c.score}%</li>`), 2)
        : "")
    : `<p class="text-xs text-white/55 mt-1">Pulsa «Analizar parcela con IA» para calcularlo.</p>`;

  cuerpo.innerHTML =
    bloque("rgba(56,189,248,0.15)", "rgba(56,189,248,0.5)", "Puntos idóneos para BALSA (agua superficial)", balsa) +
    bloque("rgba(212,175,55,0.18)", "rgba(212,175,55,0.6)", "Puntos idóneos para POZO (agua subterránea)", pozo);
  card.classList.remove("hidden");
}

function limpiarResultadoPuntos() {
  state.puntoBalsa = null;
  state.puntoPozo = null;
  state.puntosBalsaD8 = null;
  document.getElementById("card-resultado-puntos")?.classList.add("hidden");
}

// ── AGUA SUPERFICIAL CON LA U-NET (CNN) ──────────────────────────
function crearUIAguaSuperficial() {
  const ref = document.getElementById("btn-thermal");
  if (ref && !document.getElementById("btn-agua-superficial")) {
    const btn = document.createElement("button");
    btn.id = "btn-agua-superficial";
    btn.className = ref.className;
    btn.disabled = ref.disabled;
    btn.textContent = ref.disabled ? "Dibuja una parcela primero" : "💧 Detectar agua superficial (CNN)";
    ref.insertAdjacentElement("afterend", btn);
    btn.addEventListener("click", detectarAguaSuperficial);
  }
  const refCard = document.getElementById("card-thermal");
  if (refCard && !document.getElementById("card-agua")) {
    const card = document.createElement("div");
    card.id = "card-agua";
    card.className = refCard.className;
    card.classList.add("hidden");
    card.innerHTML = `
      <h3 class="card-title">💧 Agua superficial detectada (U-Net)</h3>
      <div id="metricas-agua" class="grid grid-cols-2 gap-2 mt-2"></div>
      <div id="info-agua" class="mt-2"></div>`;
    refCard.insertAdjacentElement("afterend", card);
  }
}

function overlayAgua(d) {
  const canvas = document.createElement("canvas");
  canvas.width = d.ancho;
  canvas.height = d.alto;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(d.ancho, d.alto);
  d.probabilidad_pct.forEach((p, i) => {
    const o = i * 4;
    if (p >= d.umbral_pct) {            // agua
      img.data[o] = 37; img.data[o + 1] = 99; img.data[o + 2] = 235; img.data[o + 3] = 120 + Math.round(1.35 * p);
    } else if (p >= 25) {               // dudoso
      img.data[o] = 125; img.data[o + 1] = 211; img.data[o + 2] = 252; img.data[o + 3] = 60;
    }
  });
  ctx.putImageData(img, 0, 0);
  const capa = L.imageOverlay(canvas.toDataURL(), d.bounds, { opacity: 0.9, interactive: false });
  capa.on("load", () => { const el = capa.getElement(); if (el) el.style.imageRendering = "pixelated"; });
  return capa;
}

async function detectarAguaSuperficial() {
  if (!state.ultimasParcelas) { showBanner("Dibuja primero un polígono.", "warning"); return; }
  showSpinner("Buscando agua superficial con la red neuronal U-Net... 20-40 segundos");
  try {
    const res = await apiFetch("/agua-superficial", { method: "POST", body: JSON.stringify({ coords: state.ultimasParcelas }) });
    if (!res.ok) throw new Error(await res.text());
    const d = await res.json();
    state.layerAgua.clearLayers();
    const info = document.getElementById("info-agua");
    if (!d.disponible) {
      document.getElementById("metricas-agua").innerHTML = "";
      info.innerHTML = alertHTML("warning", `⚠️ ${d.mensaje}`);
    } else {
      overlayAgua(d).addTo(state.layerAgua);
      d.masas.forEach((m, i) => {
        L.circleMarker([m.lat, m.lon], { radius: 5, color: "#1d4ed8", fillColor: "#60a5fa", fillOpacity: 1, weight: 2 })
          .bindTooltip(`💧 Masa de agua ${i + 1}: ${m.area_ha} ha (probabilidad media ${Math.round(m.probabilidad_media * 100)} %)`)
          .addTo(state.layerAgua);
      });
      document.getElementById("metricas-agua").innerHTML =
        metricCard("Agua en la parcela", `${d.area_agua_parcela_ha} ha`) +
        metricCard("Agua en el entorno", `${d.area_agua_ha} ha`) +
        metricCard("Masas de agua", d.n_masas) +
        metricCard("Calidad (IoU en test)", fmtPct(d.modelo.iou_test));
      const lista = d.masas.length
        ? d.masas.map((m, i) => `<div class="text-xs text-white/70 cursor-pointer" data-lat="${m.lat}" data-lon="${m.lon}">💧 ${i + 1}. ${m.area_ha} ha ${m.en_parcela ? "· dentro de la parcela" : "· en el entorno"} (${m.lat.toFixed(5)}, ${m.lon.toFixed(5)})</div>`).join("")
        : `<p class="text-xs text-white/60">No se ha detectado agua superficial en la parcela ni a menos de 150 m.</p>`;
      info.innerHTML = `${lista}
        <p class="text-xs text-white/50 mt-2">Azul intenso = agua (probabilidad ≥ ${d.umbral_pct} %) · azul claro = dudoso. Imagen: ${d.periodo}, ${d.resolucion_m} m por píxel.</p>
        <p class="text-xs text-white/40 mt-1">Red U-Net entrenada con etiquetas de ESA WorldCover. En test: IoU ${fmtPct(d.modelo.iou_test)}, precisión ${fmtPct(d.modelo.precision_test)}, sensibilidad ${fmtPct(d.modelo.recall_test)}.</p>
        ${d.aviso ? alertHTML("warning", `⚠️ ${d.aviso}`) : ""}`;
      info.querySelectorAll("[data-lat]").forEach(el => el.addEventListener("click", () =>
        state.mapa.setView([parseFloat(el.dataset.lat), parseFloat(el.dataset.lon)], 17)));
    }
    document.getElementById("card-agua").classList.remove("hidden");
    document.getElementById("btn-limpiar").classList.remove("hidden");
    document.getElementById("card-agua").scrollIntoView({ behavior: "smooth" });
  } catch (err) { showBanner(`❌ Error agua superficial: ${err.message}`); }
  finally { hideSpinner(); }
}

// ── ARRANQUE ─────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  initMapa();
  crearUIAguaSuperficial();

  // Mostrar nombre de usuario en cabecera
  const userInfo = document.getElementById("user-info");
  if (userInfo) {
    userInfo.textContent = `👤 ${SESSION.username} ${SESSION.role === "admin" ? "· Admin" : ""}`;
  }

  // Ocultar panel admin si no es admin
  if (SESSION.role !== "admin") {
    document.getElementById("panel-modelo-admin")?.classList.add("hidden");
  }

  fetch(`${API_BASE}/health`)
    .then(r => r.json())
    .then(d => { if (!d.gee) showBanner("⚠️ GEE no disponible.", "warning"); })
    .catch(() => showBanner("❌ Backend no responde en :8000."));

  // Logout
  document.getElementById("btn-logout")?.addEventListener("click", () => {
    localStorage.clear();
    window.location.href = "login.html";
  });
});
