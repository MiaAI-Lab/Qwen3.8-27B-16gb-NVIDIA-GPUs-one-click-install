"""A model folder without tokenizer.json is not complete (issue #8)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import downloader  # noqa: E402


def _folder(*extra: str) -> Path:
    d = Path(tempfile.mkdtemp())
    (d / "config.json").write_text("{}")
    (d / "model-00001-of-00001.safetensors").write_bytes(b"x")
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00001.safetensors"}}))
    for name in extra:
        (d / name).write_text("{}")
    return d


class TokenizerRequired(unittest.TestCase):
    def test_stopped_after_the_last_shard_is_partial(self):
        st = downloader.folder_state(_folder())
        self.assertEqual(st["state"], "partial")
        self.assertIn("tokenizer.json", st["reason"])

    def test_with_tokenizer_it_is_complete(self):
        self.assertTrue(downloader.is_complete(_folder("tokenizer.json")))

    def test_drafter_has_no_tokenizer(self):
        self.assertTrue(downloader.is_complete(_folder(), required=()))


if __name__ == "__main__":
    unittest.main()
