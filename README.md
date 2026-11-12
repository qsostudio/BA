# 💧 Buscador de Agua Subterránea — Guía de ejecución local

## 1. Estructura del proyecto

```
buscador-agua/
├── frontend/
│   ├── index.html          ← Abrir con Live Server en VS Code
│   ├── css/styles.css
│   ├── js/app.js
│   └── assets/
│       └── olivar.webp     ← Copia aquí tu imagen de fondo
│
├── backend/
│   ├── main.py             ← Entrypoint FastAPI
│   ├── services/
│   │   ├── gee_service.py  ← Google Earth Engine
│   │   └── geo_service.py  ← Open-Meteo, OpenTopoData, Macrostrat
│   ├── requirements.txt
│   ├── .env.example        ← Copia como .env y rellena
│   └── gee_key.json        ← Tu fichero de credenciales GEE (NO subir a GitHub)
│
├── .gitignore
└── README.md
```

---

## 2. Configuración de credenciales (API Keys)

### Google Earth Engine — OBLIGATORIO

| Qué necesitas          | Dónde conseguirlo                                                  | Fichero destino          |
|------------------------|--------------------------------------------------------------------|--------------------------|
| Fichero `gee_key.json` | Google Cloud Platform → IAM → Cuentas de Servicio → Crear clave (tipo JSON) | `backend/gee_key.json`   |

El fichero `.env` de backend debe contener:
```
GEE_KEY_PATH=gee_key.json
```

**Pasos para obtener gee_key.json:**
1. Ve a https://console.cloud.google.com
2. Crea o abre tu proyecto con la API de Earth Engine habilitada
3. IAM y administración → Cuentas de Servicio → clic en tu cuenta → Claves → Agregar clave → JSON
4. Descarga el JSON y cópialo como `backend/gee_key.json`
5. Asegúrate de que esa cuenta de servicio tiene el rol **Earth Engine Resource Viewer**

### APIs sin clave — todo gratuito

| Servicio       | URL                              | Clave necesaria |
|---------------|-----------------------------------|-----------------|
| Open-Meteo    | https://api.open-meteo.com        | No              |
| OpenTopoData  | https://api.opentopodata.org      | No              |
| Macrostrat    | https://macrostrat.org/api        | No              |
| Esri Imagery  | CDN público de tiles              | No              |

---

## 3. Arrancar el backend

```bash
# Situarse en la carpeta backend
cd buscador-agua/backend

# Crear entorno virtual (solo la primera vez)
python -m venv .venv

# Activar el entorno
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# Instalar dependencias (solo la primera vez)
pip install -r requirements.txt

# Copiar el fichero de entorno y configurarlo
cp .env.example .env
# Edita .env y asegúrate de que GEE_KEY_PATH apunta a tu gee_key.json

# Lanzar el servidor
uvicorn main:app --reload --port 8000
```

El backend queda disponible en: http://localhost:8000
Documentación Swagger automática: http://localhost:8000/docs

---

## 4. Arrancar el frontend con Live Server

1. Abre VS Code en la carpeta raíz `buscador-agua/`
2. Instala la extensión **Live Server** (ritwickdey.LiveServer) si no la tienes
3. Copia tu imagen `olivar.webp` en `frontend/assets/`
4. Haz clic derecho sobre `frontend/index.html` → **Open with Live Server**
5. Se abrirá el navegador en `http://127.0.0.1:5500/frontend/index.html`

> **CORS**: El backend ya tiene CORS abierto para `*` en desarrollo.
> En producción restringe `allow_origins` a tu dominio real.

---

## 5. Verificar que todo funciona

Abre http://localhost:8000/api/health en el navegador.
Debes ver: `{"status":"ok","gee":true}`

Si `"gee":false` → revisa que `gee_key.json` existe y tiene la cuenta de servicio correcta.
