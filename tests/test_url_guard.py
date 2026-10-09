import socket
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import url_guard  # noqa: E402


def resolver_for(addr):
    def r(host, port, type=0):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (addr, port))]
    return r


class CheckUrl(unittest.TestCase):
    def blocked(self, url, addr="93.184.216.34"):
        with self.assertRaises(url_guard.BlockedURL):
            url_guard.check_url(url, resolver_for(addr))

    def test_public_host_allowed(self):
        url_guard.check_url("https://example.com/a.png", resolver_for("93.184.216.34"))

    def test_loopback_private_and_metadata_refused(self):
        for addr in ("127.0.0.1", "10.0.0.5", "192.168.1.20", "172.16.0.1",
                     "169.254.169.254", "0.0.0.0", "::1", "fd00::1", "::ffff:127.0.0.1"):
            with self.subTest(addr=addr):
                self.blocked("http://anything.example/x", addr)

    def test_literal_ip_urls_are_checked_too(self):
        self.blocked("http://127.0.0.1:8888/health", "127.0.0.1")

    def test_other_schemes_refused(self):
        for u in ("file:///etc/passwd", "ftp://example.com/x", "gopher://example.com/"):
            with self.subTest(u=u):
                self.blocked(u)

    def test_one_private_answer_is_enough_to_refuse(self):
        def mixed(host, port, type=0):
            return [(socket.AF_INET, 1, 6, "", ("93.184.216.34", port)),
                    (socket.AF_INET, 1, 6, "", ("10.0.0.1", port))]
        with self.assertRaises(url_guard.BlockedURL):
            url_guard.check_url("http://rebind.example/", mixed)

    def test_opt_out_for_trusted_networks(self):
        import os
        os.environ["ALLOW_PRIVATE_IMAGE_URLS"] = "1"
        try:
            url_guard.check_url("http://10.0.0.5/x.png", resolver_for("10.0.0.5"))
        finally:
            del os.environ["ALLOW_PRIVATE_IMAGE_URLS"]


if __name__ == "__main__":
    unittest.main()
