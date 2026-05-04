import base64
import hashlib
import hmac
import os
import sqlite3
from datetime import datetime


DB_PATH = os.path.join(os.path.dirname(__file__), "users.sqlite3")
PBKDF2_ITERATIONS = 200_000


class UserAlreadyExistsError(Exception):
    pass


class InvalidCredentialsError(Exception):
    pass


def init_db(db_path=DB_PATH):
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


def create_user(username, password, db_path=DB_PATH):
    username = username.strip()
    if not username or not password:
        raise ValueError("Введіть логін і пароль.")

    init_db(db_path)
    password_hash = _hash_password(password)
    created_at = datetime.utcnow().isoformat(timespec="seconds")

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO users (username, password_hash, created_at)
                VALUES (?, ?, ?)
                """,
                (username, password_hash, created_at),
            )
            return {
                "id": cursor.lastrowid,
                "username": username,
                "face_coordinates": None,
                "created_at": created_at,
            }
    except sqlite3.IntegrityError as exc:
        raise UserAlreadyExistsError("Користувач уже існує.") from exc


def authenticate_user(username, password, db_path=DB_PATH):
    username = username.strip()
    init_db(db_path)
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
        raise InvalidCredentialsError("Неправильний логін або пароль.")

    return {
        "id": user["id"],
        "username": user["username"],
        "face_coordinates": user["face_coordinates"],
        "created_at": user["created_at"],
    }


def get_user(user_id, db_path=DB_PATH):
    init_db(db_path)
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
    return dict(user)


def update_face_coordinates(user_id, face_coordinates, db_path=DB_PATH):
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE users
            SET face_coordinates = ?
            WHERE id = ?
            """,
            (face_coordinates, user_id),
        )
