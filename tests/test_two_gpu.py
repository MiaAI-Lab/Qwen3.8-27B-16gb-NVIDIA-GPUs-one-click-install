"""Two-GPU settings: GPU_MEM_GB as a list, and the profile picker leaving it alone.

No GPU needed:  python -m unittest tests.test_two_gpu   (from the kit root)
"""
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import profiles  # noqa: E402
import serve_openai  # noqa: E402


class ParseBudgets(unittest.TestCase):
    def test_single_and_list(self):
        self.assertEqual(serve_openai.parse_budgets("14.7"), [14.7])
        self.assertEqual(serve_openai.parse_budgets("14.9,14.9"), [14.9, 14.9])
        self.assertEqual(serve_openai.parse_budgets(" 13 , 11.5 "), [13.0, 11.5])

    def test_rejects_nonsense(self):
        for bad in ("", ",", "abc", "14.7,x", "0", "-1,14", "14,,"):
            if bad == "14,,":      # trailing empties are ignored, like "14"
                self.assertEqual(serve_openai.parse_budgets(bad), [14.0])
                continue
            with self.assertRaises(SystemExit):
                serve_openai.parse_budgets(bad)


class WriteEnvKeepsList(unittest.TestCase):
    def _env(self, text):
        d = pathlib.Path(tempfile.mkdtemp())
        p = d / ".env"
        p.write_text(text, encoding="utf-8")
        return p

    def test_list_survives_the_planner(self):
        p = self._env("MODEL_DIR=models/x\nGPU_MEM_GB=14.9,14.9\nCONTEXT_SIZE=1000\n")
        profiles.write_env(p, {"GPU_MEM_GB": "14.7", "CONTEXT_SIZE": "2000", "PROFILE": "5.0"})
        t = p.read_text(encoding="utf-8")
        self.assertIn("GPU_MEM_GB=14.9,14.9", t)
        self.assertNotIn("14.7", t)
        self.assertIn("CONTEXT_SIZE=2000", t)      # the rest of the profile is still written
        self.assertIn("PROFILE=5.0", t)

    def test_quoted_list_survives(self):
        p = self._env('GPU_MEM_GB="13,13"\n')
        profiles.write_env(p, {"GPU_MEM_GB": "14.7"})
        self.assertIn("13,13", p.read_text(encoding="utf-8"))

    def test_single_value_is_still_replaced(self):
        p = self._env("GPU_MEM_GB=14.0\n")
        profiles.write_env(p, {"GPU_MEM_GB": "14.7"})
        self.assertIn("GPU_MEM_GB=14.7", p.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
