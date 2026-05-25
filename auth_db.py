import base64
import hashlib
import hmac
import os
from datetime import datetime

try:
    import psycopg
    from psycopg import errors as pg_errors
    from psycopg.rows import dict_row
except ImportError:
    psycopg = None
    pg_errors = None
    dict_row = None


BASE_DIR = os.path.dirname(__file__)
ENV_PATH = os.path.join(BASE_DIR, ".env")
DEFAULT_DATABASE_URL = "postgresql:///emotion_recognition"
PBKDF2_ITERATIONS = 200_000
PASSWORD_MIN_LENGTH = 8
PASSWORD_REQUIREMENTS_MESSAGE = (
    "Password must be at least 8 characters and use only English letters, "
    "numbers, or symbols."
)


class UserAlreadyExistsError(Exception):
    pass


class InvalidCredentialsError(Exception):
    pass


def validate_password(password):
    if len(password) < PASSWORD_MIN_LENGTH or not password.isascii():
        raise ValueError(PASSWORD_REQUIREMENTS_MESSAGE)


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


def init_db():
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


def _create_user(username, password_hash, created_at):
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


def create_user(username, password):
    username = username.strip()
    if not username or not password:
        raise ValueError("Enter a username and password.")
    validate_password(password)

    init_db()
    password_hash = _hash_password(password)
    created_at = datetime.utcnow().isoformat(timespec="seconds")

    return _create_user(username, password_hash, created_at)


def authenticate_user(username, password):
    username = username.strip()
    init_db()
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


def get_user(user_id):
    init_db()
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


def update_face_coordinates(user_id, face_coordinates):
    init_db()
    with _connect_postgres() as conn:
        conn.execute(
            """
            UPDATE users
            SET face_coordinates = %s
            WHERE id = %s
            """,
            (face_coordinates, user_id),
        )
