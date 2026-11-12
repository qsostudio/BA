"""
auth_service.py — JWT, hashing de contraseñas y gestión de usuarios.
"""

import os, uuid, random, string, datetime
from typing import Optional
import bcrypt
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from database import User

# ── Configuración ────────────────────────────────────────────────
SECRET_KEY = os.getenv("JWT_SECRET")
if not SECRET_KEY:
    raise RuntimeError("Falta JWT_SECRET en tu .env — genera uno y añádelo antes de arrancar el backend.")
ALGORITHM   = "HS256"
TOKEN_MINS  = 60 * 8   # 8 horas


# ── Contraseñas ──────────────────────────────────────────────────
def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


# ── JWT ──────────────────────────────────────────────────────────
def crear_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.datetime.utcnow() + datetime.timedelta(minutes=TOKEN_MINS)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verificar_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None


# ── CRUD de usuarios ─────────────────────────────────────────────
def obtener_usuario_por_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email.lower()).first()

def obtener_usuario_por_id(db: Session, user_id: str) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()

def crear_usuario(db: Session, email: str, username: str, password: str, role: str = "user") -> User:
    user = User(
        id              = str(uuid.uuid4()),
        email           = email.lower(),
        username        = username,
        hashed_password = hash_password(password),
        role            = role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def autenticar_usuario(db: Session, email: str, password: str) -> Optional[User]:
    user = obtener_usuario_por_email(db, email)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user

def listar_usuarios(db: Session):
    return db.query(User).all()


# ── Recuperar contraseña ─────────────────────────────────────────
def generar_codigo_reset(db: Session, email: str) -> Optional[str]:
    """
    Genera un código de 6 dígitos válido 15 minutos.
    El código se imprime en el terminal del servidor para que el admin lo comunique al usuario.
    """
    user = obtener_usuario_por_email(db, email)
    if not user:
        return None

    codigo = "".join(random.choices(string.digits, k=6))
    user.reset_code         = codigo
    user.reset_code_expires = datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
    db.commit()

    # Imprimir en el terminal del servidor (el admin lo ve)
    print("\n" + "=" * 50)
    print(f"  🔑 CÓDIGO DE RECUPERACIÓN")
    print(f"  Usuario: {email}")
    print(f"  Código:  {codigo}")
    print(f"  Válido durante 15 minutos")
    print("=" * 50 + "\n")

    return codigo

def resetear_password(db: Session, email: str, codigo: str, nueva_password: str) -> bool:
    user = obtener_usuario_por_email(db, email)
    if not user:
        return False
    if user.reset_code != codigo:
        return False
    if user.reset_code_expires < datetime.datetime.utcnow():
        return False

    user.hashed_password    = hash_password(nueva_password)
    user.reset_code         = None
    user.reset_code_expires = None
    db.commit()
    return True

def cambiar_password_admin(db: Session, user_id: str, nueva_password: str) -> bool:
    user = obtener_usuario_por_id(db, user_id)
    if not user:
        return False
    user.hashed_password = hash_password(nueva_password)
    db.commit()
    return True
