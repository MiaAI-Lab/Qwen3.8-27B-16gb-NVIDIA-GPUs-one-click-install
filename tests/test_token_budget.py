import sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import token_budget as tb  # noqa: E402


class ClampMaxTokens(unittest.TestCase):
    def test_issue_11_case_is_clamped_not_rejected(self):
        # 3.0bpw 16 GB profile: 460 pages * 256 = 117760 cache, client asks 65536,
        # prompt ~91.4k. Unclamped that is 485 pages; clamped it must fit.
        cache, prompt = 117760, 91400
        got = tb.clamp_max_tokens(prompt, 65536, cache)
        self.assertGreaterEqual(got, 1)
        pages = -(-(prompt + got) // tb.PAGE_SIZE)
        self.assertLessEqual(pages, cache // tb.PAGE_SIZE)

    def test_small_request_untouched(self):
        self.assertEqual(tb.clamp_max_tokens(1000, 512, 117760), 512)

    def test_prompt_that_cannot_fit_is_an_error(self):
        with self.assertRaises(ValueError):
            tb.clamp_max_tokens(117760, 128, 117760)

    def test_unknown_cache_leaves_request_alone(self):
        self.assertEqual(tb.clamp_max_tokens(500, 4096, None), 4096)

    def test_partial_page_is_not_usable(self):
        self.assertEqual(tb.usable_cache_tokens(1000), 768)

    def test_default_is_larger_than_the_old_1024(self):
        self.assertGreater(tb.DEFAULT_MAX_TOKENS, 1024)


if __name__ == "__main__":
    unittest.main()
