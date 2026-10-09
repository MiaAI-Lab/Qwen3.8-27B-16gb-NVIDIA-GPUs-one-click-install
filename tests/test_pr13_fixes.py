"""Regression tests for the review fixes on the Anthropic API / two-GPU port."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

try:
    import serve_openai as so  # noqa: E402  (needs aiohttp and Pillow)
except ImportError as e:      # run inside the kit's .venv to cover these
    raise unittest.SkipTest(f"serve_openai needs the server dependencies: {e}")
import dflash2  # noqa: E402
import profiles  # noqa: E402


class FakeTokenizer:
    def __init__(self):
        self.encoded = None

    def hf_render_chat_template(self, messages, **kw):
        return "A" + so.IMAGE_TRIPLE * 2 + "B"

    def hf_chat_template(self, messages, **kw):
        raise AssertionError("not used")

    def encode(self, text, **kw):
        assert "embeddings" not in kw, "count_only must not embed"
        self.encoded = text

        class Ids:
            shape = (1, len(text.split("<|image_pad|>")) - 1)
        return Ids()


class CountOnly(unittest.TestCase):
    def test_images_are_estimated_not_fetched_or_embedded(self):
        saved = dict(so.vision)
        try:
            so.vision["model"] = object()      # would blow up if it were used
            so.vision["max_pixels"] = 1024 * 1024
            tok = FakeTokenizer()
            msgs = [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "http://127.0.0.1:1/x.png"}},
                {"type": "image_url", "image_url": {"url": "http://10.0.0.1/y.png"}},
            ]}]
            ids, emb = so.build_inputs(tok, msgs, None, True, None, count_only = True)
            self.assertIsNone(emb)
            self.assertEqual(tok.encoded.count("<|image_pad|>"), 2 * 1024)
        finally:
            so.vision.clear(); so.vision.update(saved)


class Images(unittest.TestCase):
    def test_too_many_images_refused(self):
        part = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        msgs = [{"role": "user", "content": [part] * (so.MAX_IMAGES + 1)}]
        with self.assertRaises(ValueError):
            so.extract_images(msgs)

    def test_internal_urls_refused_without_saying_why(self):
        for url in ("http://127.0.0.1:8888/health", "http://169.254.169.254/latest/meta-data/",
                    "http://192.168.1.1/"):
            with self.subTest(url = url):
                with self.assertRaises(ValueError) as cm:
                    so.decode_image(url)
                msg = str(cm.exception)
                self.assertIn("not allowed", msg)
                self.assertNotIn("Connection", msg)
                self.assertNotIn("refused", msg)


class ToolResultIsError(unittest.TestCase):
    def test_error_flag_reaches_the_model(self):
        body = {"model": "m", "max_tokens": 16, "messages": [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "run", "input": {}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "is_error": True,
                 "content": "command not found"}]},
        ]}
        openai, err = so.anthropic_to_openai(body)
        self.assertIsNone(err)
        tool = next(m for m in openai["messages"] if m["role"] == "tool")
        self.assertTrue(tool["content"].startswith("Error:"))
        self.assertIn("command not found", tool["content"])


class TwoGpuEnv(unittest.TestCase):
    def test_profile_planner_keeps_a_per_gpu_list(self):
        d = Path(tempfile.mkdtemp()); p = d / ".env"
        p.write_text("GPU_MEM_GB=14.9,7.2\nCONTEXT_SIZE=100000\n")
        profiles.write_env(p, {"GPU_MEM_GB": "14.7", "CONTEXT_SIZE": "262144"})
        text = p.read_text()
        self.assertIn("GPU_MEM_GB=14.9,7.2", text)
        self.assertIn("CONTEXT_SIZE=262144", text)

    def test_single_value_is_still_updated(self):
        d = Path(tempfile.mkdtemp()); p = d / ".env"
        p.write_text("GPU_MEM_GB=20\n")
        profiles.write_env(p, {"GPU_MEM_GB": "14.7"})
        self.assertIn("GPU_MEM_GB=14.7", p.read_text())

    def test_dflash2_refused_on_a_split(self):
        self.assertTrue(dflash2.multi_gpu_message("14.9,7.2"))
        self.assertEqual(dflash2.multi_gpu_message("22.8"), "")


if __name__ == "__main__":
    unittest.main()


class HttpSmoke(unittest.IsolatedAsyncioTestCase):
    """count_tokens must answer while a generation holds the GPU lock."""

    async def asyncSetUp(self):
        import threading
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer
        self.saved = dict(so.vision)
        so.vision["model"] = object()
        so.vision["max_pixels"] = 1024 * 1024
        app = web.Application()
        app["generator"], app["tokenizer"] = object(), FakeTokenizer()
        app["max_body_mb"] = 8
        app.router.add_post("/v1/messages/count_tokens", so.anthropic_count_tokens)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()
        self.lock_held = threading.Event()
        self.release = threading.Event()

        def hold():
            with so.gen_lock:
                self.lock_held.set()
                self.release.wait(10)
        self.t = threading.Thread(target = hold, daemon = True)
        self.t.start()
        self.lock_held.wait(5)

    async def asyncTearDown(self):
        self.release.set()
        self.t.join(5)
        await self.client.close()
        so.vision.clear(); so.vision.update(self.saved)

    async def test_count_tokens_with_images_does_not_wait_for_generation(self):
        import asyncio
        body = {"model": "m", "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "url", "url": "http://10.0.0.9/a.png"}},
            {"type": "text", "text": "hi"}]}]}
        r = await asyncio.wait_for(
            self.client.post("/v1/messages/count_tokens", json = body), timeout = 3)
        self.assertEqual(r.status, 200, await r.text())
        self.assertIn("input_tokens", await r.json())
