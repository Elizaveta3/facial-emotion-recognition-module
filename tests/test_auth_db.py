import uuid
import unittest

from auth_db import (
    InvalidCredentialsError,
    PASSWORD_REQUIREMENTS_MESSAGE,
    UserAlreadyExistsError,
    _connect_postgres,
    _hash_password,
    _verify_password,
    authenticate_user,
    create_user,
    get_user,
    init_db,
    psycopg,
    update_face_coordinates,
    validate_password,
)


class AuthDbTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        unavailable_errors = (RuntimeError,)
        if psycopg is not None:
            unavailable_errors = (RuntimeError, psycopg.OperationalError)

        try:
            init_db()
        except unavailable_errors as exc:
            raise unittest.SkipTest(f"PostgreSQL is not available: {exc}") from exc

    def setUp(self):
        self.username_prefix = f"test_user_{uuid.uuid4().hex}"

    def tearDown(self):
        with _connect_postgres() as conn:
            conn.execute(
                "DELETE FROM users WHERE username LIKE %s",
                (f"{self.username_prefix}%",),
            )

    def username(self, suffix=""):
        return f"{self.username_prefix}{suffix}"

    def test_create_user_trims_username_and_authenticates_password(self):
        username = self.username()
        user = create_user(f"  {username}  ", "secret123")

        self.assertEqual(user["username"], username)
        self.assertIsNone(user["face_coordinates"])
        authenticated = authenticate_user(username, "secret123")
        self.assertEqual(authenticated["id"], user["id"])
        self.assertEqual(authenticated["username"], username)

    def test_create_user_rejects_empty_username_or_password(self):
        with self.assertRaises(ValueError):
            create_user("   ", "secret123")

        with self.assertRaises(ValueError):
            create_user(self.username(), "")

    def test_create_user_rejects_short_or_non_english_password(self):
        with self.assertRaisesRegex(ValueError, PASSWORD_REQUIREMENTS_MESSAGE):
            create_user(self.username("_short"), "secret")

        with self.assertRaisesRegex(ValueError, PASSWORD_REQUIREMENTS_MESSAGE):
            create_user(self.username("_unicode"), "пароль123")

        validate_password("secret123")

    def test_create_user_rejects_duplicate_username(self):
        username = self.username()
        create_user(username, "secret123")

        with self.assertRaises(UserAlreadyExistsError):
            create_user(username, "another1")

    def test_authenticate_user_rejects_wrong_password_and_missing_user(self):
        username = self.username()
        create_user(username, "secret123")

        with self.assertRaises(InvalidCredentialsError):
            authenticate_user(username, "wrongpass")

        with self.assertRaises(InvalidCredentialsError):
            authenticate_user(self.username("_missing"), "secret123")

    def test_update_face_coordinates_and_get_user(self):
        user = create_user(self.username(), "secret123")

        update_face_coordinates(user["id"], '{"baseline": {}}')
        stored = get_user(user["id"])

        self.assertEqual(stored["face_coordinates"], '{"baseline": {}}')
        self.assertIsNone(get_user(-1))

    def test_password_hash_uses_salt_and_does_not_store_plain_password(self):
        first = _hash_password("secret")
        second = _hash_password("secret")

        self.assertNotEqual(first, second)
        self.assertNotIn("secret", first)
        self.assertTrue(_verify_password("secret", first))
        self.assertFalse(_verify_password("wrong", first))
        self.assertFalse(_verify_password("secret", "invalid-format"))

    def test_init_db_is_idempotent(self):
        init_db()
        init_db()

        with _connect_postgres() as conn:
            table = conn.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = 'users'
                """
            ).fetchone()

        self.assertIsNotNone(table)


if __name__ == "__main__":
    unittest.main()
