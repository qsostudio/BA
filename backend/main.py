from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from typing import List, Optional
from io import BytesIO
import asyncio
import math
import os
import time
import traceback
import uuid
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
load_dotenv()

# ── Base de datos ────────────────────────────────────────────────
from database import crear_tablas, get_db, SessionLocal
from sqlalchemy.orm import Session

# ── Servicios ────────────────────────────────────────────────────
from services.geo_service import obtener_humedad, obtener_elevacion, obtener_geologia, clasificar_geologia
from services.profundidad_service import estimar_profundidad
from services.gee_service import init_gee, get_ndvi_punto, analizar_parcela_gee, get_ndvi_serie, get_thermal_anomalias
from services.ign_service import calcular_vaguadas
from services.modelo_ia   import (entrenar, predecir, cargar_metricas, modelo_disponible,
                                  elevacion_srtm_gee, GEOLOGIA_A_SCORE)
from services.unet_service import unet_disponible, detectar_agua_superficial, cargar_metricas_unet
from services.pdf_service import generar_informe_pdf
from services.pozo_profundo_service import encontrar_punto_pozo_profundo
from services.pozos_reales_service import pozos_en_area
from services.auth_service import (
    crear_usuario, autenticar_usuario, obtener_usuario_por_email,
    obtener_usuario_por_id, crear_token, verificar_token,
    generar_codigo_reset, resetear_password, listar_usuarios,
    cambiar_password_admin
)

# ── App ──────────────────────────────────────────────────────────
app = FastAPI(title="Buscador de Agua — API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)
GEE_OK   = False


@app.on_event("startup")
async def startup_event():
    global GEE_OK
    crear_tablas()
    GEE_OK = init_gee()
    print(f"[GEE] Inicializado: {GEE_OK}")


# ── Helpers de autenticación ─────────────────────────────────────
def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    if not credentials:
        raise HTTPException(status_code=401, detail="Token requerido")
    payload = verificar_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Token inválido o expirado")
    user = obtener_usuario_por_id(db, payload.get("sub"))
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="Usuario no encontrado")
    return user

def get_admin_user(user=Depends(get_current_user)):
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Se requiere rol de administrador")
    return user


# ── Modelos de request ───────────────────────────────────────────
class PuntoRequest(BaseModel):
    latitud:  float
    longitud: float

class ParcelaRequest(BaseModel):
    coords: List[List[float]]

class ProfundidadRequest(BaseModel):
    coords:   Optional[List[List[float]]] = None   # poligono [[lon, lat], ...] -> se usa su centroide
    latitud:  Optional[float] = None               # o un punto concreto (p. ej. la estrella dorada)
    longitud: Optional[float] = None

class AreaRequest(BaseModel):
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

class RegisterRequest(BaseModel):
    email:    str
    username: str
    password: str

class LoginRequest(BaseModel):
    email:    str
    password: str

class ForgotPasswordRequest(BaseModel):
    email: str

class ResetPasswordRequest(BaseModel):
    email:         str
    codigo:        str
    nueva_password: str

class CambiarPasswordAdmin(BaseModel):
    user_id:       str
    nueva_password: str


# ────────────────────────────────────────────────────────────────
# AUTH ENDPOINTS
# ────────────────────────────────────────────────────────────────

@app.post("/api/auth/register")
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if obtener_usuario_por_email(db, req.email):
        raise HTTPException(status_code=400, detail="El email ya está registrado")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="La contraseña debe tener al menos 6 caracteres")
    user  = crear_usuario(db, req.email, req.username, req.password, role="user")
    token = crear_token({"sub": user.id, "role": user.role, "username": user.username})
    return {"token": token, "role": user.role, "username": user.username, "email": user.email}


@app.post("/api/auth/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = autenticar_usuario(db, req.email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Email o contraseña incorrectos")
    token = crear_token({"sub": user.id, "role": user.role, "username": user.username})
    return {"token": token, "role": user.role, "username": user.username, "email": user.email}


@app.get("/api/auth/me")
def me(user=Depends(get_current_user)):
    return {"id": user.id, "email": user.email, "username": user.username, "role": user.role}


@app.post("/api/auth/forgot-password")
def forgot_password(req: ForgotPasswordRequest, db: Session = Depends(get_db)):
    generar_codigo_reset(db, req.email)
    # Siempre devolvemos OK para no revelar si el email existe
    return {"ok": True, "mensaje": "Si el email existe, el administrador verá el código en el servidor."}


@app.post("/api/auth/reset-password")
def reset_password(req: ResetPasswordRequest, db: Session = Depends(get_db)):
    ok = resetear_password(db, req.email, req.codigo, req.nueva_password)
    if not ok:
        raise HTTPException(status_code=400, detail="Código incorrecto, expirado o email no encontrado")
    return {"ok": True, "mensaje": "Contraseña cambiada correctamente"}


# ── Admin: gestión de usuarios ───────────────────────────────────
@app.get("/api/admin/usuarios")
def admin_usuarios(admin=Depends(get_admin_user), db: Session = Depends(get_db)):
    usuarios = listar_usuarios(db)
    return [{"id": u.id, "email": u.email, "username": u.username,
             "role": u.role, "is_active": u.is_active,
             "created_at": str(u.created_at)} for u in usuarios]


@app.post("/api/admin/cambiar-password")
def admin_cambiar_password(req: CambiarPasswordAdmin, admin=Depends(get_admin_user), db: Session = Depends(get_db)):
    ok = cambiar_password_admin(db, req.user_id, req.nueva_password)
    if not ok:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    return {"ok": True}


# ────────────────────────────────────────────────────────────────
# ANÁLISIS ENDPOINTS (requieren login)
# ────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "gee": GEE_OK,
            "modelo_profundidad": modelo_disponible(), "unet_agua": unet_disponible()}


@app.post("/api/analizar-punto")
async def analizar_punto(req: PuntoRequest, user=Depends(get_current_user)):
    try:
        humedad   = await obtener_humedad(req.latitud, req.longitud)
        elevacion = await obtener_elevacion(req.latitud, req.longitud)
        geo       = await obtener_geologia(req.latitud, req.longitud)
        roca_es, puntuacion_geo, estado_geo = clasificar_geologia(geo.get("lith", ""))
        puntuacion_base = min(60, int(humedad * 1000 + max(0, 500 - elevacion) / 10))
        puntuacion      = min(100, puntuacion_base + puntuacion_geo)
        ndvi_value = None
        if GEE_OK:
            ndvi_value = get_ndvi_punto(req.latitud, req.longitud)
        profundidad = await estimar_profundidad(req.latitud, req.longitud)
        return {
            "humedad": humedad, "elevacion": elevacion, "puntuacion": puntuacion,
            "roca": roca_es, "descripcion_roca": geo.get("descrip", "Sin descripción"),
            "nombre_unidad": geo.get("name", "Desconocido"),
            "estado_geo": estado_geo, "ndvi": ndvi_value,
            "profundidad": profundidad,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analizar-parcela")
async def analizar_parcela(req: ParcelaRequest, user=Depends(get_current_user)):
    if not GEE_OK:
        raise HTTPException(status_code=503, detail="Google Earth Engine no disponible")
    try:
        resultado = analizar_parcela_gee(req.coords)
        lats = [c[1] for c in req.coords]
        lons = [c[0] for c in req.coords]
        lat_c = sum(lats) / len(lats)
        lon_c = sum(lons) / len(lons)
        geo = await obtener_geologia(lat_c, lon_c)
        roca_es, _, estado_geo = clasificar_geologia(geo.get("lith", ""))
        resultado["geologia"] = {
            "roca": roca_es, "nombre_unidad": geo.get("name", "Desconocido"),
            "descripcion": geo.get("descrip", "Sin descripción"), "estado": estado_geo,
        }
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/vaguadas")
async def vaguadas(req: ParcelaRequest, user=Depends(get_current_user)):
    try:
        resultado = await calcular_vaguadas(req.coords)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Chequeo de acuífero (geología real + pozos reales del IGME cercanos)
    # añadido al mismo resultado -- en un try/except APARTE, para que si
    # GEE no está disponible o falla, las vaguadas (que no dependen de GEE)
    # sigan funcionando exactamente igual que antes.
    try:
        if GEE_OK:
            resultado["pozo_acuifero"] = await encontrar_punto_pozo_profundo(req.coords)
        else:
            resultado["pozo_acuifero"] = {"disponible": False, "mensaje": "Google Earth Engine no disponible"}
    except Exception:
        resultado["pozo_acuifero"] = {"disponible": False, "mensaje": "No se pudo calcular en este momento"}

    return resultado


@app.post("/api/punto-pozo-profundo")
async def punto_pozo_profundo(req: ParcelaRequest, user=Depends(get_current_user)):
    """
    Punto óptimo para POZO PROFUNDO (agua subterránea) — distinto del
    punto óptimo de /analizar-parcela, que es de "balsa" (acumulación
    superficial). No modifica /analizar-parcela en absoluto: es un
    endpoint totalmente independiente.
    """
    if not GEE_OK:
        raise HTTPException(status_code=503, detail="Google Earth Engine no disponible")
    try:
        return await encontrar_punto_pozo_profundo(req.coords)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/pozos-reales-area")
def pozos_reales_area(req: AreaRequest, user=Depends(get_current_user)):
    """
    Pozos reales del IGME (dataset local, ya cargado por el usuario) dentro
    de una caja de coordenadas — pensado para pintarlos en el mapa según
    la vista actual. No depende de ningún servidor externo.
    """
    try:
        pozos = pozos_en_area(req.lat_min, req.lat_max, req.lon_min, req.lon_max)
        return {"pozos": pozos, "n_total": len(pozos)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _generar_pdf_bytes(coords):
    """Mismos cálculos que antes; las partes bloqueantes van en un hilo aparte
    para no congelar el servidor mientras se genera el informe."""
    resultado_gee      = await asyncio.to_thread(analizar_parcela_gee, coords)
    resultado_vaguadas = await calcular_vaguadas(coords)
    resultado_pozo     = await encontrar_punto_pozo_profundo(coords)

    lats = [c[1] for c in coords]
    lons = [c[0] for c in coords]
    lat_c, lon_c = sum(lats) / len(lats), sum(lons) / len(lons)
    geo = await obtener_geologia(lat_c, lon_c)
    roca_es, _, estado_geo = clasificar_geologia(geo.get("lith", ""))
    geologia = {
        "roca": roca_es, "nombre_unidad": geo.get("name", "Desconocido"),
        "descripcion": geo.get("descrip", "Sin descripción"), "estado": estado_geo,
    }
    resultado_ia = await _resultados_ia(coords, resultado_pozo)
    return await asyncio.to_thread(generar_informe_pdf, coords, resultado_gee, resultado_vaguadas, geologia,
                                   resultado_pozo, resultado_ia)


@app.post("/api/informe-pdf")
async def informe_pdf(req: ParcelaRequest, user=Depends(get_current_user)):
    """
    Genera el informe PDF completo de la parcela: reutiliza EXACTAMENTE
    los mismos cálculos que /analizar-parcela, /vaguadas, /punto-pozo-profundo
    y la geología (mismas funciones, mismos parámetros), para que el PDF
    explique con fidelidad lo que ya se ve en la app — no un cálculo aparte.
    """
    if not GEE_OK:
        raise HTTPException(status_code=503, detail="Google Earth Engine no disponible")
    try:
        pdf_bytes = await _generar_pdf_bytes(req.coords)
        return StreamingResponse(
            BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=informe_agua_subterranea.pdf"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ndvi-serie")
async def ndvi_serie(req: ParcelaRequest, user=Depends(get_current_user)):
    if not GEE_OK:
        raise HTTPException(status_code=503, detail="Google Earth Engine no disponible")
    try:
        return get_ndvi_serie(req.coords)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ────────────────────────────────────────────────────────────────
# TAREAS LARGAS EN SEGUNDO PLANO (informe PDF y NDVI de 4 años)
# Tardan más de un minuto: si se esperan en una sola petición, la conexión
# se puede cortar por el camino (por ejemplo, en Codespaces). Se lanzan en
# segundo plano y el navegador pregunta cada pocos segundos si han acabado.
# ────────────────────────────────────────────────────────────────
_TRABAJOS = {}


def _lanzar_trabajo(tipo, corrutina):
    ahora = time.time()
    for k in [k for k, t in _TRABAJOS.items() if ahora - t["inicio"] > 3600]:
        _TRABAJOS.pop(k, None)
    tid = uuid.uuid4().hex
    trabajo = {"tipo": tipo, "estado": "en_curso", "inicio": ahora}
    _TRABAJOS[tid] = trabajo

    async def correr():
        try:
            trabajo["resultado"] = await corrutina
            trabajo["estado"] = "ok"
        except HTTPException as e:
            trabajo["estado"], trabajo["error"] = "error", str(e.detail)
        except Exception as e:
            traceback.print_exc()
            trabajo["estado"], trabajo["error"] = "error", f"{type(e).__name__}: {e}"

    trabajo["tarea"] = asyncio.create_task(correr())
    return tid


@app.post("/api/trabajos/informe-pdf")
async def trabajo_informe_pdf(req: ParcelaRequest, user=Depends(get_current_user)):
    if not GEE_OK:
        raise HTTPException(status_code=503, detail="Google Earth Engine no disponible")
    return {"id": _lanzar_trabajo("pdf", _generar_pdf_bytes(req.coords))}


@app.post("/api/trabajos/ndvi-serie")
async def trabajo_ndvi_serie(req: ParcelaRequest, user=Depends(get_current_user)):
    if not GEE_OK:
        raise HTTPException(status_code=503, detail="Google Earth Engine no disponible")
    return {"id": _lanzar_trabajo("ndvi", asyncio.to_thread(get_ndvi_serie, req.coords))}


@app.get("/api/trabajos/{tid}")
async def estado_trabajo(tid: str, user=Depends(get_current_user)):
    t = _TRABAJOS.get(tid)
    if t is None:
        raise HTTPException(status_code=404, detail="Tarea no encontrada (¿se ha reiniciado el servidor?). Vuelve a pulsar el botón.")
    r = {"estado": t["estado"], "segundos": round(time.time() - t["inicio"])}
    if t["estado"] == "error":
        r["error"] = t["error"]
    elif t["estado"] == "ok" and t["tipo"] != "pdf":
        r["resultado"] = t["resultado"]
    return r


@app.get("/api/trabajos/{tid}/pdf")
async def descargar_pdf_trabajo(tid: str, user=Depends(get_current_user)):
    t = _TRABAJOS.get(tid)
    if t is None or t["tipo"] != "pdf" or t["estado"] != "ok":
        raise HTTPException(status_code=404, detail="El informe no está listo.")
    return Response(content=t["resultado"], media_type="application/pdf",
                    headers={"Content-Disposition": "attachment; filename=informe_agua_subterranea.pdf"})


@app.post("/api/thermal")
async def thermal(req: ParcelaRequest, user=Depends(get_current_user)):
    if not GEE_OK:
        raise HTTPException(status_code=503, detail="Google Earth Engine no disponible")
    try:
        return get_thermal_anomalias(req.coords)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ────────────────────────────────────────────────────────────────
# MODELO IA — entrenar solo admin, predecir todos
# ────────────────────────────────────────────────────────────────

@app.get("/api/modelo-info")
def modelo_info(user=Depends(get_current_user)):
    if not modelo_disponible():
        return {"disponible": False, "mensaje": "Modelo no entrenado todavía.", "unet": cargar_metricas_unet()}
    return {"disponible": True, "metricas": cargar_metricas(), "unet": cargar_metricas_unet()}


@app.post("/api/entrenar")
async def entrenar_modelo(
    file: UploadFile = File(...),
    admin=Depends(get_admin_user)   # ← solo admin
):
    import tempfile, shutil
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        metricas = entrenar(tmp_path)
        return {"ok": True, "metricas": metricas}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/predecir-modelo")
async def predecir_modelo(req: dict, user=Depends(get_current_user)):
    try:
        return predecir(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ────────────────────────────────────────────────────────────────
# PROFUNDIDAD DEL AGUA SUBTERRÁNEA (Gradient Boosting + intervalo)
# ────────────────────────────────────────────────────────────────
@app.post("/api/profundidad")
async def profundidad(req: ProfundidadRequest, user=Depends(get_current_user)):
    """
    Estima la profundidad del agua subterránea en un punto (o en el centroide
    de la parcela) con su intervalo de error. Las variables se calculan con
    las MISMAS funciones que en el entrenamiento (entrenar_profundidad.py):
    elevación SRTM 30 m en GEE, geología clasificada con clasificar_geologia
    e IDW de los pozos del IGME guardados con el modelo.
    """
    if req.latitud is not None and req.longitud is not None:
        lat, lon = req.latitud, req.longitud
    elif req.coords:
        lat = sum(c[1] for c in req.coords) / len(req.coords)
        lon = sum(c[0] for c in req.coords) / len(req.coords)
    else:
        raise HTTPException(status_code=400, detail="Indica un punto (latitud, longitud) o una parcela (coords).")
    return await _profundidad_en_punto(lat, lon)


async def _profundidad_en_punto(lat, lon):
    """Profundidad estimada con su intervalo en un punto. La usan /api/profundidad
    y el informe PDF, para que los dos den exactamente lo mismo."""
    elevacion, fuente_elevacion = None, None
    if GEE_OK:
        try:
            valor = float((await asyncio.to_thread(elevacion_srtm_gee, [lat], [lon]))[0])
            if not math.isnan(valor):
                elevacion, fuente_elevacion = valor, "SRTM 30 m (Google Earth Engine)"
        except Exception:
            pass
    if elevacion is None:
        try:
            elevacion = float(await obtener_elevacion(lat, lon))
            fuente_elevacion = "OpenTopoData (respaldo)"
        except Exception:
            pass

    geologia_score, estado_geo = None, None
    try:
        geo = await obtener_geologia(lat, lon)
        _, _, estado_geo = clasificar_geologia(geo.get("lith", ""))
        geologia_score = GEOLOGIA_A_SCORE.get(estado_geo)
    except Exception:
        pass

    resultado = predecir({"lat": lat, "lon": lon, "elevacion": elevacion, "geologia_score": geologia_score})
    resultado["punto"] = {"lat": lat, "lon": lon}
    resultado["geologia"] = estado_geo
    resultado["fuente_elevacion"] = fuente_elevacion
    return resultado


async def _resultados_ia(coords, resultado_pozo):
    """Lo que calculan los dos modelos entrenados, para el informe PDF."""
    ia = {"metricas_profundidad": cargar_metricas(), "metricas_unet": cargar_metricas_unet()}
    try:
        if resultado_pozo and resultado_pozo.get("disponible"):
            lat, lon = resultado_pozo["pozo_profundo_lat"], resultado_pozo["pozo_profundo_lon"]
            sitio = "estrella dorada (punto POZO)"
        else:
            lat = sum(c[1] for c in coords) / len(coords)
            lon = sum(c[0] for c in coords) / len(coords)
            sitio = "centro de la parcela"
        ia["profundidad"] = await _profundidad_en_punto(lat, lon)
        ia["profundidad"]["sitio"] = sitio
    except Exception as e:
        traceback.print_exc()
        ia["profundidad"] = {"disponible": False, "mensaje": f"no se pudo estimar la profundidad ({e})"}
    try:
        ia["agua"] = (await asyncio.to_thread(detectar_agua_superficial, coords) if GEE_OK
                      else {"disponible": False, "mensaje": "Google Earth Engine no disponible"})
    except Exception as e:
        traceback.print_exc()
        ia["agua"] = {"disponible": False, "mensaje": f"no se pudo detectar el agua superficial ({e})"}
    return ia


# ────────────────────────────────────────────────────────────────
# AGUA SUPERFICIAL (U-Net): balsas, charcas y embalses
# ────────────────────────────────────────────────────────────────
@app.post("/api/agua-superficial")
async def agua_superficial(req: ParcelaRequest, user=Depends(get_current_user)):
    if not GEE_OK:
        raise HTTPException(status_code=503, detail="Google Earth Engine no disponible")
    try:
        return detectar_agua_superficial(req.coords)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.mount("/app", StaticFiles(directory="../frontend", html=True), name="frontend")