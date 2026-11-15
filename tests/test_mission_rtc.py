"""Mission RTC option: validated, defaults off, follows the selected model in the web remote."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mission  # noqa: E402


class MissionRtcOptionTests(unittest.TestCase):
    def test_default_off_and_bool_only(self):
        self.assertFalse(mission.options({})["rtc"])
        self.assertTrue(mission.options({"rtc": True})["rtc"])
        with self.assertRaises(ValueError):
            mission.options({"rtc": "yes"})

    def test_rtc_wrapper_detection(self):
        import web_move
        self.assertTrue(web_move.model_rtc({"command": ["/root/vla_grasp_v3_step13000_rtc.sh"]}))
        self.assertFalse(web_move.model_rtc({"command": ["/root/vla_grasp_v3_step13000.sh"]}))
        self.assertFalse(web_move.model_rtc({"command": ["/root/x_rtc.sh"], "rtc": False}))


if __name__ == "__main__":
    unittest.main()
