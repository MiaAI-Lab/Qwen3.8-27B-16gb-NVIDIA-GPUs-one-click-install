"""Which URLs the server may fetch on a client's behalf (image_url parts).

The server is often reachable from the LAN and has no authentication, so an
image URL is a request the *server* makes with the *server's* network position.
Only public http(s) hosts are allowed: every address the name resolves to must
be globally routable (no loopback, private, link-local - which covers cloud
metadata services - multicast or reserved ranges), and every redirect hop is
checked again. Errors are deliberately uninformative to the caller so the
endpoint cannot be used to map the internal network.

Residual: the name is resolved here and again by the HTTP client, so a DNS
server that answers differently the second time can still win. Closing that
needs connecting by address. Set ALLOW_PRIVATE_IMAGE_URLS=1 to switch the check
off for a trusted network (an image host on your LAN).

Standard library only.
"""
from __future__ import annotations

import ipaddress
import os
import socket
import urllib.parse
import urllib.request

MAX_REDIRECTS = 3


class BlockedURL(ValueError):
    pass


def _allowed(addr: str) -> bool:
    try:
        ip = ipaddress.ip_address(addr.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def check_url(url: str, resolver=socket.getaddrinfo) -> None:
    """Raise BlockedURL unless `url` is http(s) and resolves only to public addresses."""
    if os.environ.get("ALLOW_PRIVATE_IMAGE_URLS", "").strip().lower() in ("1", "true", "yes"):
        return
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise BlockedURL("not an http(s) URL")
    try:
        infos = resolver(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80),
                         type=socket.SOCK_STREAM)
    except OSError as e:
        raise BlockedURL("host did not resolve") from e
    addrs = {i[4][0] for i in infos}
    if not addrs or not all(_allowed(a) for a in addrs):
        raise BlockedURL("host is not a public address")


class _GuardedRedirect(urllib.request.HTTPRedirectHandler):
    max_redirections = MAX_REDIRECTS

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(urllib.parse.urljoin(req.full_url, newurl))
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch(url: str, *, timeout: float = 20, limit: int = 48 * 1024 * 1024,
          headers: dict | None = None) -> bytes:
    """GET a public URL; at most `limit` bytes. Raises BlockedURL / OSError / ValueError."""
    check_url(url)
    opener = urllib.request.build_opener(_GuardedRedirect)
    req = urllib.request.Request(url, headers = headers or {})
    with opener.open(req, timeout = timeout) as r:
        raw = r.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f"larger than {limit // (1024 * 1024)} MB")
    return raw
