import sqlite3

from auth_db import DB_PATH, _connect_postgres, init_db


def migrate_users(sqlite_path=DB_PATH):
    init_db()
    with sqlite3.connect(sqlite_path) as sqlite_conn:
        sqlite_conn.row_factory = sqlite3.Row
        users = sqlite_conn.execute(
            """
            SELECT username, password_hash, face_coordinates, created_at
            FROM users
            ORDER BY id
            """
        ).fetchall()

    inserted = 0
    with _connect_postgres() as pg_conn:
        with pg_conn.cursor() as cursor:
            for user in users:
                cursor.execute(
                    """
                    INSERT INTO users (
                        username,
                        password_hash,
                        face_coordinates,
                        created_at
                    )
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (username) DO NOTHING
                    RETURNING id
                    """,
                    (
                        user["username"],
                        user["password_hash"],
                        user["face_coordinates"],
                        user["created_at"],
                    ),
                )
                if cursor.fetchone() is not None:
                    inserted += 1
    return inserted, len(users)


if __name__ == "__main__":
    inserted_count, total_count = migrate_users()
    print(f"Migrated {inserted_count} of {total_count} users.")
