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


_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_V4_COMPAT = ipaddress.ip_network("::/96")
_SIIT = ipaddress.ip_network("::ffff:0:0:0/96")      # IPv4-translated (RFC 2765)


def _embedded_v4(ip: ipaddress.IPv6Address):
    """The IPv4 address a v6 address carries, if it carries one (mapped, 6to4,
    Teredo, NAT64, deprecated IPv4-compatible); otherwise None."""
    if ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    if ip.sixtofour is not None:
        return ip.sixtofour
    if ip.teredo is not None:
        return ip.teredo[1]
    if ip in _NAT64 or ip in _V4_COMPAT or ip in _SIIT:
        return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    return None


def _allowed(addr: str) -> bool:
    try:
        ip = ipaddress.ip_address(addr.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        v4 = _embedded_v4(ip)
        if v4 is not None:
            return _allowed(str(v4))
    return ip.is_global and not ip.is_multicast


def check_url(url: str, resolver=socket.getaddrinfo) -> None:
    """Raise BlockedURL unless `url` is http(s) and resolves only to public addresses."""
    if os.environ.get("ALLOW_PRIVATE_IMAGE_URLS", "").strip().lower() in ("1", "true", "yes"):
        return
    if "\\" in url or any(ord(c) < 33 for c in url):
        raise BlockedURL("malformed URL")
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise BlockedURL("not an http(s) URL")
    if parts.username is not None or parts.password is not None:
        raise BlockedURL("credentials in the URL")
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
          headers: dict | None = None, deadline: float = 30.0) -> bytes:
    """GET a public URL; at most `limit` bytes and `deadline` seconds in total
    (`timeout` is per socket operation, which a host that drips bytes never
    trips). Raises BlockedURL / OSError / ValueError."""
    import time
    check_url(url)
    end = time.monotonic() + deadline
    opener = urllib.request.build_opener(_GuardedRedirect)
    req = urllib.request.Request(url, headers = headers or {})
    chunks, size = [], 0
    with opener.open(req, timeout = timeout) as r:
        while True:
            chunk = r.read(64 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                raise ValueError(f"larger than {limit // (1024 * 1024)} MB")
            if time.monotonic() > end:
                raise TimeoutError("fetch took too long")
            chunks.append(chunk)
    return b"".join(chunks)
