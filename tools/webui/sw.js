/* Simplex service worker.
 *
 * Its only job is to make the installed app open instantly and survive the
 * server being restarted for a model switch. It caches the four static files
 * the shell is made of and nothing else: every /ui/* call, /v1, /health and
 * the event streams go straight to the network, because a cached answer there
 * would be a stale conversation or a stalled stream.
 */
const VERSION = "simplex-shell-v1";
const SHELL = [
  "/",
  "/ui/static/style.css",
  "/ui/static/app.js",
  "/icon-192.png",
  "/icon-512.png",
];

self.addEventListener("install", (e) => {
  e.waitUntil(
    caches.open(VERSION)
      .then((c) => c.addAll(SHELL))
      .then(() => self.skipWaiting())
      .catch(() => self.skipWaiting())   // offline first install: not fatal
  );
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

function isShell(url) {
  return url.pathname === "/"
    || url.pathname.startsWith("/ui/static/")
    || /^\/(icon-|apple-touch-icon)/.test(url.pathname);
}

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin !== location.origin) return;
  if (!isShell(url)) return;               // API and streams: never intercepted

  /* Network first, so a rebuilt UI shows up on the next load; the cache is the
     fallback for "the server is restarting into another model". */
  e.respondWith(
    fetch(req)
      .then((res) => {
        if (res && res.ok) {
          const copy = res.clone();
          caches.open(VERSION).then((c) => c.put(req, copy)).catch(() => {});
        }
        return res;
      })
      .catch(() => caches.match(req).then((hit) => hit || caches.match("/")))
  );
});
