import base64
import hashlib
import hmac
import os
import sqlite3
from datetime import datetime

try:
    import psycopg
    from psycopg import errors as pg_errors
    from psycopg.rows import dict_row
except ImportError:
    psycopg = None
    pg_errors = None
    dict_row = None


DB_PATH = os.path.join(os.path.dirname(__file__), "users.sqlite3")
BASE_DIR = os.path.dirname(__file__)
ENV_PATH = os.path.join(BASE_DIR, ".env")
DEFAULT_DATABASE_URL = "postgresql:///emotion_recognition"
PBKDF2_ITERATIONS = 200_000


class UserAlreadyExistsError(Exception):
    pass


class InvalidCredentialsError(Exception):
    pass


def _load_env_value(key, env_path=ENV_PATH):
    if not os.path.exists(env_path):
        return None

    with open(env_path, encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == key:
                return value.strip().strip("\"'")
    return None


def _get_database_url():
    return os.getenv("DATABASE_URL") or _load_env_value("DATABASE_URL") or DEFAULT_DATABASE_URL


def _connect_postgres():
    if psycopg is None:
        raise RuntimeError(
            "PostgreSQL driver is not installed. Run: pip install 'psycopg[binary]'"
        )
    return psycopg.connect(_get_database_url(), row_factory=dict_row)


def _init_sqlite(db_path):
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                face_coordinates TEXT,
                created_at TEXT NOT NULL
            )
            """
        )


def _init_postgres():
    with _connect_postgres() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                face_coordinates TEXT,
                created_at TEXT NOT NULL
            )
            """
        )


def init_db(db_path=None):
    if db_path is not None:
        _init_sqlite(db_path)
    else:
        _init_postgres()


def _hash_password(password, salt=None):
    if salt is None:
        salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    salt_b64 = base64.b64encode(salt).decode("ascii")
    digest_b64 = base64.b64encode(digest).decode("ascii")
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt_b64}${digest_b64}"


def _verify_password(password, stored_hash):
    try:
        algorithm, iterations, salt_b64, digest_b64 = stored_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        salt = base64.b64decode(salt_b64.encode("ascii"))
        expected = base64.b64decode(digest_b64.encode("ascii"))
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            int(iterations),
        )
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def _create_user_sqlite(username, db_path, password_hash, created_at):
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO users (username, password_hash, created_at)
                VALUES (?, ?, ?)
                """,
                (username, password_hash, created_at),
            )
            user_id = cursor.lastrowid
    except sqlite3.IntegrityError as exc:
        raise UserAlreadyExistsError("User already exists.") from exc

    return {
        "id": user_id,
        "username": username,
        "face_coordinates": None,
        "created_at": created_at,
    }


def _create_user_postgres(username, password_hash, created_at):
    try:
        with _connect_postgres() as conn:
            user = conn.execute(
                """
                INSERT INTO users (username, password_hash, created_at)
                VALUES (%s, %s, %s)
                RETURNING id, username, face_coordinates, created_at
                """,
                (username, password_hash, created_at),
            ).fetchone()
    except pg_errors.UniqueViolation as exc:
        raise UserAlreadyExistsError("User already exists.") from exc

    return dict(user)


def create_user(username, password, db_path=None):
    username = username.strip()
    if not username or not password:
        raise ValueError("Enter a username and password.")

    init_db(db_path)
    password_hash = _hash_password(password)
    created_at = datetime.utcnow().isoformat(timespec="seconds")

    if db_path is not None:
        return _create_user_sqlite(username, db_path, password_hash, created_at)
    return _create_user_postgres(username, password_hash, created_at)


def _authenticate_user_sqlite(username, password, db_path):
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        user = conn.execute(
            """
            SELECT id, username, password_hash, face_coordinates, created_at
            FROM users
            WHERE username = ?
            """,
            (username,),
        ).fetchone()

    if user is None or not _verify_password(password, user["password_hash"]):
        raise InvalidCredentialsError("Incorrect username or password.")

    return {
        "id": user["id"],
        "username": user["username"],
        "face_coordinates": user["face_coordinates"],
        "created_at": user["created_at"],
    }


def _authenticate_user_postgres(username, password):
    with _connect_postgres() as conn:
        user = conn.execute(
            """
            SELECT id, username, password_hash, face_coordinates, created_at
            FROM users
            WHERE username = %s
            """,
            (username,),
        ).fetchone()

    if user is None or not _verify_password(password, user["password_hash"]):
        raise InvalidCredentialsError("Incorrect username or password.")

    return {
        "id": user["id"],
        "username": user["username"],
        "face_coordinates": user["face_coordinates"],
        "created_at": user["created_at"],
    }


def authenticate_user(username, password, db_path=None):
    username = username.strip()
    init_db(db_path)
    if db_path is not None:
        return _authenticate_user_sqlite(username, password, db_path)
    return _authenticate_user_postgres(username, password)


def _get_user_sqlite(user_id, db_path):
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        user = conn.execute(
            """
            SELECT id, username, face_coordinates, created_at
            FROM users
            WHERE id = ?
            """,
            (user_id,),
        ).fetchone()
    if user is None:
        return None
    return {
        "id": user["id"],
        "username": user["username"],
        "face_coordinates": user["face_coordinates"],
        "created_at": user["created_at"],
    }


def _get_user_postgres(user_id):
    with _connect_postgres() as conn:
        user = conn.execute(
            """
            SELECT id, username, face_coordinates, created_at
            FROM users
            WHERE id = %s
            """,
            (user_id,),
        ).fetchone()
    if user is None:
        return None
    return dict(user)


def get_user(user_id, db_path=None):
    init_db(db_path)
    if db_path is not None:
        return _get_user_sqlite(user_id, db_path)
    return _get_user_postgres(user_id)


def _update_face_coordinates_sqlite(user_id, face_coordinates, db_path):
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE users
            SET face_coordinates = ?
            WHERE id = ?
            """,
            (face_coordinates, user_id),
        )


def _update_face_coordinates_postgres(user_id, face_coordinates):
    with _connect_postgres() as conn:
        conn.execute(
            """
            UPDATE users
            SET face_coordinates = %s
            WHERE id = %s
            """,
            (face_coordinates, user_id),
        )


def update_face_coordinates(user_id, face_coordinates, db_path=None):
    init_db(db_path)
    if db_path is not None:
        _update_face_coordinates_sqlite(user_id, face_coordinates, db_path)
    else:
        _update_face_coordinates_postgres(user_id, face_coordinates)
