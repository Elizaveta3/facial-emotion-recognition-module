import unittest
from unittest.mock import Mock, patch

from main import EmotionRecognitionApp
from recognition import CAMERA_UNAVAILABLE_MESSAGE, CameraUnavailableError


class FakeApp:
    def __init__(self):
        self.config = Mock()
        self.update_idletasks = Mock()


class MainErrorHandlingTest(unittest.TestCase):
    def test_run_busy_notifies_user_when_camera_is_unavailable(self):
        app = FakeApp()
        on_done = Mock()

        def action():
            raise CameraUnavailableError(CAMERA_UNAVAILABLE_MESSAGE)

        with patch("main.messagebox.showerror") as showerror:
            EmotionRecognitionApp._run_busy(app, action, on_done)

        showerror.assert_called_once_with("Error", CAMERA_UNAVAILABLE_MESSAGE)
        app.config.assert_any_call(cursor="watch")
        app.config.assert_any_call(cursor="")
        on_done.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
