import os
import sqlite3
import tempfile
import unittest

from auth_db import (
    InvalidCredentialsError,
    UserAlreadyExistsError,
    _hash_password,
    _verify_password,
    authenticate_user,
    create_user,
    get_user,
    init_db,
    update_face_coordinates,
)


class AuthDbTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "users.sqlite3")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_create_user_trims_username_and_authenticates_password(self):
        user = create_user("  natalia  ", "secret", db_path=self.db_path)

        self.assertEqual(user["username"], "natalia")
        self.assertIsNone(user["face_coordinates"])
        authenticated = authenticate_user("natalia", "secret", db_path=self.db_path)
        self.assertEqual(authenticated["id"], user["id"])
        self.assertEqual(authenticated["username"], "natalia")

    def test_create_user_rejects_empty_username_or_password(self):
        with self.assertRaises(ValueError):
            create_user("   ", "secret", db_path=self.db_path)

        with self.assertRaises(ValueError):
            create_user("natalia", "", db_path=self.db_path)

    def test_create_user_rejects_duplicate_username(self):
        create_user("natalia", "secret", db_path=self.db_path)

        with self.assertRaises(UserAlreadyExistsError):
            create_user("natalia", "another", db_path=self.db_path)

    def test_authenticate_user_rejects_wrong_password_and_missing_user(self):
        create_user("natalia", "secret", db_path=self.db_path)

        with self.assertRaises(InvalidCredentialsError):
            authenticate_user("natalia", "wrong", db_path=self.db_path)

        with self.assertRaises(InvalidCredentialsError):
            authenticate_user("missing", "secret", db_path=self.db_path)

    def test_update_face_coordinates_and_get_user(self):
        user = create_user("natalia", "secret", db_path=self.db_path)

        update_face_coordinates(user["id"], '{"baseline": {}}', db_path=self.db_path)
        stored = get_user(user["id"], db_path=self.db_path)

        self.assertEqual(stored["face_coordinates"], '{"baseline": {}}')
        self.assertIsNone(get_user(999, db_path=self.db_path))

    def test_password_hash_uses_salt_and_does_not_store_plain_password(self):
        first = _hash_password("secret")
        second = _hash_password("secret")

        self.assertNotEqual(first, second)
        self.assertNotIn("secret", first)
        self.assertTrue(_verify_password("secret", first))
        self.assertFalse(_verify_password("wrong", first))
        self.assertFalse(_verify_password("secret", "invalid-format"))

    def test_init_db_is_idempotent(self):
        init_db(self.db_path)
        init_db(self.db_path)

        with sqlite3.connect(self.db_path) as conn:
            table = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'users'"
            ).fetchone()

        self.assertIsNotNone(table)


if __name__ == "__main__":
    unittest.main()
