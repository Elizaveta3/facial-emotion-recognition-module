import json
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/facial_emotion_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/facial_emotion_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

from recognition import (
    CAMERA_UNAVAILABLE_MESSAGE,
    CameraUnavailableError,
    _open_camera,
    parse_face_profile,
    serialize_face_profile,
    _safe_session_owner,
)


class FakeClosedCamera:
    def __init__(self):
        self.released = False

    def isOpened(self):
        return False

    def release(self):
        self.released = True


class RecognitionUtilsTest(unittest.TestCase):
    def test_parse_face_profile_returns_none_for_empty_or_invalid_payload(self):
        self.assertIsNone(parse_face_profile(""))
        self.assertIsNone(parse_face_profile(None))
        self.assertIsNone(parse_face_profile("{invalid"))
        self.assertIsNone(parse_face_profile("[1, 2, 3]"))

    def test_parse_and_serialize_face_profile_round_trip_unicode_payload(self):
        profile = {
            "baseline": {"ear_avg": 0.3},
            "owner": "Наталія",
        }

        serialized = serialize_face_profile(profile)
        parsed = parse_face_profile(serialized)

        self.assertEqual(parsed, profile)
        self.assertEqual(json.loads(serialized)["owner"], "Наталія")

    def test_safe_session_owner_removes_unsafe_filename_characters(self):
        self.assertEqual(_safe_session_owner(" nat/test user! "), "nat_test_user")
        self.assertEqual(_safe_session_owner("..."), "...")
        self.assertEqual(_safe_session_owner("!!!"), "user")
        self.assertIsNone(_safe_session_owner(""))
        self.assertIsNone(_safe_session_owner(None))

    def test_open_camera_raises_clear_error_when_camera_is_unavailable(self):
        camera = FakeClosedCamera()

        with patch("recognition.cv2.VideoCapture", return_value=camera):
            with self.assertRaisesRegex(CameraUnavailableError, CAMERA_UNAVAILABLE_MESSAGE):
                _open_camera()

        self.assertTrue(camera.released)


if __name__ == "__main__":
    unittest.main()
