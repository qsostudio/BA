"""
crear_admin.py — Crea la primera cuenta de administrador.

Ejecutar UNA sola vez desde la carpeta backend/:
    python crear_admin.py

Puedes cambiar el email y contraseña antes de ejecutarlo.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from database import crear_tablas, SessionLocal
from services.auth_service import crear_usuario, obtener_usuario_por_email

# ── Configura aquí tus credenciales de admin ─────────────────────
import getpass

ADMIN_EMAIL    = os.getenv("ADMIN_EMAIL") or input("Email del admin a crear: ").strip()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD") or getpass.getpass("Contraseña del admin a crear: ")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME") or input("Nombre de usuario del admin a crear: ").strip()
# ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  CREAR CUENTA DE ADMINISTRADOR")
    print("=" * 50)

    crear_tablas()
    db = SessionLocal()

    existente = obtener_usuario_por_email(db, ADMIN_EMAIL)
    if existente:
        print(f"\n  ⚠️  Ya existe un usuario con el email {ADMIN_EMAIL}")
        print("  Si quieres cambiar la contraseña, edita este script y\n  cambia ADMIN_EMAIL por uno nuevo.")
        db.close()
        return

    usuario = crear_usuario(db, ADMIN_EMAIL, ADMIN_USERNAME, ADMIN_PASSWORD, role="admin")
    db.close()

    print(f"\n  ✅ Administrador creado correctamente")
    print(f"\n  Email:      {ADMIN_EMAIL}")
    print(f"  Usuario:    {ADMIN_USERNAME}")
    print(f"  Contraseña: {ADMIN_PASSWORD}")
    print(f"  Rol:        admin")
    print(f"\n  ⚠️  Guarda estas credenciales en un lugar seguro.")
    print("=" * 50)

if __name__ == "__main__":
    main()
