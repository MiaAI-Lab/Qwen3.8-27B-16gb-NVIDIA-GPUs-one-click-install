"""Resumable Hugging Face downloader - standard library only.

Why not huggingface_hub: this runs *before* the virtualenv exists, so the
weights can come down while pip is still building the engine. It also gives
us byte-level progress, a real ETA, and a resume that survives a crash or a
closed lid, none of which snapshot_download reports back to a web page.

Layout on disk is the same one snapshot_download produces with local_dir=...:
plain files under `dest`, so the rest of the kit does not care which of the
two fetched them. Partial files live next to their target as `<name>.part`
with a sidecar `<name>.part.json` holding the size/etag we resumed against;
a `.part` whose sidecar does not match the file on the server is discarded
rather than silently concatenated onto the wrong bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
USER_AGENT = "simplex-kit/1.0 (resumable downloader)"
CHUNK = 1024 * 1024
TIMEOUT = 60
SKIP_NAMES = {".gitattributes"}


class DownloadError(RuntimeError):
    def __init__(self, message: str, hint: str = "", retryable: bool = False):
        super().__init__(message)
        self.hint = hint
        # A 503 from a CDN is worth another go; a 404 is not. Without this the
        # retry loop below never sees an HTTP failure at all, because _open has
        # already translated it out of the URLError family.
        self.retryable = retryable


@dataclass
class RemoteFile:
    path: str
    size: int
    oid: str = ""          # sha256 for LFS files, git blob sha otherwise
    lfs: bool = False


@dataclass
class Progress:
    """One immutable-ish snapshot. The setup page renders exactly this."""
    total_bytes: int = 0
    done_bytes: int = 0
    files_total: int = 0
    files_done: int = 0
    current: str = ""
    speed_bps: float = 0.0
    state: str = "idle"      # idle | listing | downloading | verifying | done | error | cancelled
    message: str = ""

    @property
    def eta_seconds(self) -> float:
        left = max(0, self.total_bytes - self.done_bytes)
        if self.speed_bps <= 1:
            return -1.0
        return left / self.speed_bps

    def as_dict(self) -> dict:
        pct = 0.0
        if self.total_bytes > 0:
            pct = min(100.0, 100.0 * self.done_bytes / self.total_bytes)
        return {
            "total_bytes": self.total_bytes,
            "done_bytes": self.done_bytes,
            "percent": round(pct, 2),
            "files_total": self.files_total,
            "files_done": self.files_done,
            "current": self.current,
            "speed_bps": round(self.speed_bps),
            "eta_seconds": round(self.eta_seconds, 1),
            "state": self.state,
            "message": self.message,
        }


class _SafeRedirects(urllib.request.HTTPRedirectHandler):
    """urllib copies every header except the content ones onto a redirect,
    including Authorization - and Hugging Face redirects each file to a CDN on
    a different host. Sending the user's token there would hand a third party
    read (often write) access to their private repositories, so it is dropped
    the moment the host changes. Non-HTTP schemes are refused outright."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlparse(newurl)
        if parsed.scheme not in ("http", "https"):
            raise DownloadError(
                f"The download was redirected to an address Simplex will not follow "
                f"({parsed.scheme or 'no'} scheme).",
                "That is not something a normal Hugging Face download does.")
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is None:
            return None
        if _host_of(newurl) != _host_of(req.full_url):
            for name in ("Authorization", "authorization"):
                new.headers.pop(name, None)
                new.unredirected_hdrs.pop(name, None)
        return new


def _host_of(url: str) -> str:
    p = urllib.parse.urlparse(url)
    return (p.hostname or "").lower()


_OPENER = urllib.request.build_opener(_SafeRedirects)


def _request(url: str, token: str = "", start: int = 0) -> urllib.request.Request:
    req = urllib.request.Request(url)
    req.add_header("User-Agent", USER_AGENT)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if start > 0:
        req.add_header("Range", f"bytes={start}-")
    return req


def _open(url: str, token: str = "", start: int = 0, timeout: int = TIMEOUT):
    try:
        return _OPENER.open(_request(url, token, start), timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise DownloadError(
                f"Hugging Face refused the download ({e.code}).",
                "The repository is gated or private. Accept its licence on huggingface.co "
                "and put an access token in HF_TOKEN.",
            ) from e
        if e.code == 404:
            raise DownloadError(
                f"Not found on Hugging Face: {url}",
                "Check HF_TARGET_REPO and HF_REVISION in .env.",
            ) from e
        if e.code == 416:      # asked to resume past the end of the file
            raise DownloadError("resume-range-rejected") from e
        transient = e.code in (408, 425, 429, 500, 502, 503, 504)
        raise DownloadError(
            f"Hugging Face returned HTTP {e.code}.",
            "That is usually a temporary problem at their end - try again in a minute."
            if transient else "",
            retryable=transient) from e
    except urllib.error.URLError as e:
        raise DownloadError(
            f"Could not reach {HF_ENDPOINT} ({e.reason}).",
            "Check the internet connection, or a proxy/firewall blocking huggingface.co.",
            retryable=True,
        ) from e


_BAD_SEGMENTS = {"", ".", ".."}


def safe_relpath(path: str) -> str | None:
    """A repository file path we are willing to write, or None.

    Everything in the tree listing is chosen by whoever published the repo, so
    it is untrusted input that ends up in a filesystem path. `dest / "../../x"`
    escapes the models folder, and on Windows `dest / "C:/Users/Public/x.bat"`
    discards `dest` entirely - which, with a Startup folder one directory
    traversal away, is code execution. Only plain relative segments pass."""
    if not path or len(path) > 1024:
        return None
    if "\\" in path:            # a Windows separator has no business in this API
        return None
    if path.startswith("/") or path.startswith("~"):
        return None
    if re.match(r"^[A-Za-z]:", path):        # drive letter, absolute on Windows
        return None
    parts = path.split("/")
    if any(seg in _BAD_SEGMENTS for seg in parts):
        return None
    if any(seg.endswith((" ", ".")) or set(seg) & set('<>:"|?*') or
           any(ord(c) < 32 for c in seg) for seg in parts):
        return None              # names Windows cannot represent, or control chars
    return "/".join(parts)


def _contained(dest: Path, target: Path) -> bool:
    """Belt to safe_relpath's braces: the resolved target must still be under
    dest once the operating system has had its say about the path."""
    try:
        target.resolve().relative_to(dest.resolve())
        return True
    except (ValueError, OSError):
        return False


def list_files(repo: str, revision: str = "", token: str = "") -> list[RemoteFile]:
    """The repo's file tree. Folders are walked; anything not a file is dropped."""
    rev = urllib.parse.quote(revision or "main", safe="")
    url = f"{HF_ENDPOINT}/api/models/{repo}/tree/{rev}?recursive=1&expand=1"
    with _open(url, token) as r:
        data = json.loads(r.read().decode("utf-8"))
    out: list[RemoteFile] = []
    for entry in data:
        if entry.get("type") != "file":
            continue
        path = safe_relpath(entry.get("path") or "")
        if not path or path.rsplit("/", 1)[-1] in SKIP_NAMES:
            continue
        lfs = entry.get("lfs") or None
        if lfs:
            out.append(RemoteFile(path, int(lfs.get("size") or entry.get("size") or 0),
                                  str(lfs.get("oid") or lfs.get("sha256") or ""), True))
        else:
            out.append(RemoteFile(path, int(entry.get("size") or 0), str(entry.get("oid") or ""), False))
    if not out:
        raise DownloadError(
            f"{repo} has no files at revision {revision or 'main'}.",
            "Check HF_REVISION - EXL3 quants live on branches such as 4.00bpw.",
        )
    return out


def _sidecar(part: Path) -> Path:
    return part.with_suffix(part.suffix + ".json")


def _resume_from(part: Path, remote: RemoteFile) -> int:
    """Bytes already on disk we are allowed to keep. 0 means start over."""
    if not part.is_file():
        return 0
    meta_path = _sidecar(part)
    try:
        meta = json.loads(meta_path.read_text("utf-8"))
    except Exception:                      # noqa: BLE001 - no sidecar, no trust
        return 0
    if meta.get("size") != remote.size or meta.get("oid") != remote.oid:
        return 0
    have = part.stat().st_size
    if have <= 0 or have > remote.size:
        return 0
    # `have == remote.size` is a download that finished and then died before the
    # rename - a closed lid at 99%. Keeping it saves re-fetching several GB; the
    # checksum is what decides whether it was really complete.
    return have


def _verify(target: Path, remote: RemoteFile) -> None:
    if remote.size and target.stat().st_size != remote.size:
        raise DownloadError(
            f"{remote.path} came down the wrong size "
            f"({target.stat().st_size} of {remote.size} bytes).",
            "Delete the file and run the download again.",
        )
    if not (remote.lfs and len(remote.oid) == 64):
        return
    h = hashlib.sha256()
    with target.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    if h.hexdigest() != remote.oid:
        target.unlink(missing_ok=True)
        raise DownloadError(
            f"{remote.path} failed its checksum and was deleted.",
            "That is usually a truncated or proxied download - run it again.",
        )


class Download:
    """One repo -> one folder. Thread-safe progress; cancel() stops promptly."""

    def __init__(self, repo: str, dest: Path, revision: str = "", token: str = "",
                 on_progress=None, retries: int = 4):
        self.repo, self.dest = repo, Path(dest)
        self.revision, self.token = revision or "", token or ""
        self.retries = max(1, retries)
        self._on_progress = on_progress
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._last_emit = 0.0
        self._window: list[tuple[float, int]] = []
        self.progress = Progress()

    # ---------------------------------------------------------------- state --
    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def _emit(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_emit < 0.4:
            return
        self._last_emit = now
        if self._on_progress:
            try:
                self._on_progress(self.progress.as_dict())
            except Exception:              # noqa: BLE001 - a listener must never break a download
                pass

    def _advance(self, n: int) -> None:
        now = time.time()
        with self._lock:
            self.progress.done_bytes += n
            self._window.append((now, n))
            cut = now - 8.0
            while self._window and self._window[0][0] < cut:
                self._window.pop(0)
            if len(self._window) >= 2:
                span = max(0.25, self._window[-1][0] - self._window[0][0])
                self.progress.speed_bps = sum(b for _, b in self._window) / span
        self._emit()

    # ------------------------------------------------------------------ run --
    def run(self) -> Path:
        """Fetch everything. Raises DownloadError; `progress.state` is always
        left at a terminal value first, so nothing that watches it can hang."""
        try:
            return self._run()
        except BaseException as e:          # noqa: BLE001 - re-raised immediately
            with self._lock:
                if self.progress.state not in ("done", "cancelled", "error"):
                    self.progress.state = "cancelled" if self.cancelled else "error"
                    self.progress.message = str(e)
            self._emit(force=True)
            raise

    def _run(self) -> Path:
        p = self.progress
        p.state, p.message = "listing", "Asking Hugging Face what this model contains"
        self._emit(force=True)
        files = list_files(self.repo, self.revision, self.token)

        self.dest.mkdir(parents=True, exist_ok=True)
        pending: list[RemoteFile] = []
        already = 0
        for f in files:
            target = self.dest / f.path
            if not _contained(self.dest, target):
                raise DownloadError(
                    f"{self.repo} lists a file that would be written outside the model "
                    f"folder ({f.path}).",
                    "That is not something a normal model repository does. Nothing was "
                    "written; check HF_TARGET_REPO in .env.")
            if target.is_file() and (not f.size or target.stat().st_size == f.size):
                already += f.size
                continue
            pending.append(f)

        with self._lock:
            p.files_total = len(files)
            p.files_done = len(files) - len(pending)
            p.total_bytes = sum(f.size for f in files)
            p.done_bytes = already
            p.state = "downloading"
            p.message = ""
        self._emit(force=True)

        if not pending:
            p.state, p.message = "done", "Weights were already on disk"
            self._emit(force=True)
            return self.dest

        for f in pending:
            if self.cancelled:
                p.state, p.message = "cancelled", "Download stopped"
                self._emit(force=True)
                return self.dest
            with self._lock:
                p.current = f.path
            self._emit(force=True)
            try:
                self._fetch(f)
            except BaseException as e:      # noqa: BLE001 - re-raised below
                # `state` has to reach a terminal value before the exception
                # leaves, or every watcher polling it (setup_core's cancel
                # thread, the page's progress bar) waits for a download that
                # is never coming back.
                with self._lock:
                    p.state = "cancelled" if self.cancelled else "error"
                    p.message = str(e)
                self._emit(force=True)
                raise
            with self._lock:
                p.files_done += 1
            self._emit(force=True)

        with self._lock:
            p.current, p.state = "", "done"
            p.message = "Weights downloaded"
            p.done_bytes = max(p.done_bytes, p.total_bytes)
        self._emit(force=True)
        return self.dest

    def _fetch(self, f: RemoteFile) -> None:
        target = self.dest / f.path
        if not _contained(self.dest, target):
            raise DownloadError(
                f"{self.repo} lists a file that would be written outside the model "
                f"folder ({f.path}).",
                "That is not something a normal model repository does. Nothing was "
                "written; check HF_TARGET_REPO in .env.")
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        url = f"{HF_ENDPOINT}/{self.repo}/resolve/{urllib.parse.quote(self.revision or 'main', safe='')}/{urllib.parse.quote(f.path)}"

        last: Exception | None = None
        # How many of this file's bytes are already in done_bytes. It is always
        # read back from the file on disk rather than from what _stream claims,
        # because a stream that failed halfway still wrote (and counted) what it
        # managed - and the old code, which only updated this on success, then
        # counted that prefix a second time on the retry.
        counted = 0

        def on_disk() -> int:
            try:
                return part.stat().st_size if part.is_file() else 0
            except OSError:
                return 0

        for attempt in range(self.retries):
            if self.cancelled:
                return
            start = _resume_from(part, f)
            if start == 0:
                part.unlink(missing_ok=True)
                _sidecar(part).unlink(missing_ok=True)
            if start > counted:
                # a part file left by an earlier run, or by the attempt before
                # this one: real progress that has not been counted yet
                self._advance(start - counted)
                counted = start
            elif start < counted:
                with self._lock:
                    self.progress.done_bytes = max(0, self.progress.done_bytes - (counted - start))
                counted = start
            if f.size and start >= f.size:
                break                     # finished last time, only the rename is left
            try:
                self._stream(url, part, f, start)
                counted = max(counted, on_disk())
                break
            except DownloadError as e:
                counted = max(counted, on_disk())
                if str(e) == "resume-range-rejected":
                    part.unlink(missing_ok=True)
                    _sidecar(part).unlink(missing_ok=True)
                    with self._lock:
                        self.progress.done_bytes = max(0, self.progress.done_bytes - counted)
                    counted = 0
                    last = e
                    continue
                if not e.retryable or attempt == self.retries - 1:
                    raise
                last = e
            except (OSError, urllib.error.URLError) as e:
                counted = max(counted, on_disk())
                last = e
                if attempt == self.retries - 1:
                    break
            # give the network a moment; the next pass resumes where it stopped
            for _ in range(int(2 ** attempt * 2)):
                if self.cancelled:
                    return
                time.sleep(0.5)
        else:
            last = last or RuntimeError("download did not start")

        if self.cancelled:
            return
        if not part.is_file() or (f.size and part.stat().st_size != f.size):
            raise DownloadError(
                f"{f.path} did not finish downloading"
                + (f" ({last})" if last else "") + ".",
                "Start the kit again - the download picks up where it stopped.",
            )
        os.replace(part, target)
        _sidecar(part).unlink(missing_ok=True)
        self.progress.state = "verifying"
        self._emit(force=True)
        _verify(target, f)
        self.progress.state = "downloading"

    def _stream(self, url: str, part: Path, f: RemoteFile, start: int) -> None:
        """Write `part` from byte `start`. How much arrived is read back off the
        disk by the caller, not returned from here, so a stream that dies
        halfway is accounted for exactly like one that succeeded."""
        resp = _open(url, self.token, start)
        try:
            if start and getattr(resp, "status", 200) != 206:
                # server ignored Range: take it from the top rather than append
                with self._lock:
                    self.progress.done_bytes = max(0, self.progress.done_bytes - start)
                start = 0
            mode = "ab" if start else "wb"
            _sidecar(part).write_text(
                json.dumps({"size": f.size, "oid": f.oid, "url": url}), encoding="utf-8")
            with part.open(mode) as out:
                while True:
                    if self.cancelled:
                        return
                    block = resp.read(CHUNK)
                    if not block:
                        break
                    out.write(block)
                    self._advance(len(block))
        finally:
            resp.close()


def humanize(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def is_complete(dest: Path) -> bool:
    """What the launcher checks before deciding a download is needed at all."""
    dest = Path(dest)
    return (dest / "config.json").is_file() and any(dest.glob("*.safetensors")) \
        and not any(dest.glob("*.part"))


def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Resumable Hugging Face download (no dependencies)")
    ap.add_argument("repo")
    ap.add_argument("dest")
    ap.add_argument("--revision", default="")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN", ""))
    a = ap.parse_args(argv)

    def show(p: dict) -> None:
        eta = p["eta_seconds"]
        eta_s = "--:--" if eta < 0 else f"{int(eta) // 60:02d}:{int(eta) % 60:02d}"
        print(f"\r  {p['percent']:5.1f}%  {humanize(p['done_bytes'])} / {humanize(p['total_bytes'])}"
              f"  {humanize(p['speed_bps'])}/s  ETA {eta_s}  {p['current'][:34]:<34}",
              end="", flush=True)

    dl = Download(a.repo, Path(a.dest), a.revision, a.token, on_progress=show)
    try:
        dl.run()
    except DownloadError as e:
        print(f"\n  {e}\n  {e.hint}")
        return 1
    except KeyboardInterrupt:
        dl.cancel()
        print("\n  Stopped - run this again to resume.")
        return 130
    print("\n  Done.")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
