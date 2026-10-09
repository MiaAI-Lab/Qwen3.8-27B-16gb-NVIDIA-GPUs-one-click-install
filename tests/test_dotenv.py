"""The Windows readers and the profile writer must keep a hash the shell keeps."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from cli import read_env
from profiles import read_env as profile_read_env
from profiles import write_env
from win_start import load_dotenv
from dsh import env_value


def _write(text: str) -> Path:
    directory = Path(tempfile.mkdtemp())
    path = directory / ".env"
    path.write_text(text, encoding="utf-8")
    return path


class DotenvHash(unittest.TestCase):
    def test_quoted_hash_survives(self):
        path = _write('HF_TOKEN="hf_abc#def"\n')
        for reader in (load_dotenv, profile_read_env, read_env):
            with self.subTest(reader=reader.__module__):
                self.assertEqual(reader(path)["HF_TOKEN"], "hf_abc#def")

    def test_hash_without_a_space_is_part_of_the_value(self):
        path = _write("NAME=hello#world\n")
        for reader in (load_dotenv, profile_read_env, read_env):
            with self.subTest(reader=reader.__module__):
                self.assertEqual(reader(path)["NAME"], "hello#world")

    def test_spaced_hash_is_still_a_comment(self):
        path = _write("NAME=hello # world\n")
        for reader in (load_dotenv, profile_read_env, read_env):
            with self.subTest(reader=reader.__module__):
                self.assertEqual(reader(path)["NAME"], "hello")

    def test_writer_round_trip(self):
        path = _write("OTHER=1\n")
        write_env(path, {"HF_TOKEN": "hf_abc#def", "CONTEXT_SIZE": "262144"})
        stored = path.read_text(encoding="utf-8")
        self.assertIn('HF_TOKEN="hf_abc#def"', stored)
        self.assertIn("CONTEXT_SIZE=262144", stored)
        self.assertEqual(load_dotenv(path)["HF_TOKEN"], "hf_abc#def")
        self.assertEqual(profile_read_env(path)["HF_TOKEN"], "hf_abc#def")
        self.assertEqual(read_env(path)["CONTEXT_SIZE"], "262144")


class DshEnvValue(unittest.TestCase):
    """The harness launcher read API_KEY with its own parser and cut it at '#'."""

    def test_quoted_hash_kept(self):
        self.assertEqual(env_value('"key#1"'), "key#1")

    def test_trailing_comment_removed(self):
        self.assertEqual(env_value("abc   # note"), "abc")

    def test_hash_without_space_kept(self):
        self.assertEqual(env_value("abc#def"), "abc#def")


if __name__ == "__main__":
    unittest.main()
