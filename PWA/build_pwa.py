#!/usr/bin/env python3
"""
build_pwa.py — Builds the FeedlyVP PWA into docs/

Reads:  digest_log.json  (project root)
        index.html        (project root, template)

Writes: docs/index.html   (built app with data injected)
        docs/sw.js        (service worker)
        docs/manifest.json
        docs/icons/icon-192.png
        docs/icons/icon-512.png
        docs/.nojekyll

Run:  python build_pwa.py
"""

import json
import struct
import zlib
from pathlib import Path
from datetime import datetime, timezone

# ── Paths ────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.resolve()          # → .../PWA/
REPO_ROOT  = ROOT.parent                              # → .../FeedlyVP/
TEMPLATE   = ROOT / "index.html"                      # → PWA/index.html
LOG        = REPO_ROOT / "digest_log.json"            # → FeedlyVP/digest_log.json
DOCS       = ROOT / "docs"                            # → PWA/docs/
ICONS_DIR  = DOCS / "icons"                           # → PWA/docs/icons/

DOCS.mkdir(exist_ok=True)
ICONS_DIR.mkdir(exist_ok=True)


# ── Digest data ──────────────────────────────────────────────
def load_data() -> list:
    if not LOG.exists():
        print("  ℹ  digest_log.json not found — using empty data")
        return []
    with open(LOG, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        return []
    # Sort newest-first; normalise article keys for safety
    entries = sorted(raw, key=lambda x: x.get("date", ""), reverse=True)
    print(f"  ✓  Loaded {len(entries)} digest entries "
          f"(latest: {entries[0]['date'] if entries else 'none'})")
    return entries


# ── PNG icon (pure stdlib) ───────────────────────────────────
def make_png(size: int, r: int, g: int, b: int) -> bytes:
    """Create a solid-colour PNG using only stdlib (no Pillow needed)."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        body   = tag + data
        crc    = struct.pack(">I", zlib.crc32(body) & 0xFFFF_FFFF)
        return length + body + crc

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    row  = b"\x00" + bytes([r, g, b]) * size          # filter byte + pixels
    idat = zlib.compress(row * size, level=9)

    png  = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", ihdr)
    png += chunk(b"IDAT", idat)
    png += chunk(b"IEND", b"")
    return png


# ── Individual file writers ──────────────────────────────────
def write_nojekyll():
    path = DOCS / ".nojekyll"
    path.write_text("")
    print("  ✓  .nojekyll")


def write_icons():
    # Navy: #1a2744 → (26, 39, 68)
    r, g, b = 0x1A, 0x27, 0x44
    for size in (192, 512):
        (ICONS_DIR / f"icon-{size}.png").write_bytes(make_png(size, r, g, b))
    print("  ✓  icons/icon-192.png  icons/icon-512.png")


def write_manifest():
    manifest = {
        "name":             "FeedlyVP",
        "short_name":       "FeedlyVP",
        "description":      "Your personal tech news digest",
        "theme_color":      "#1a2744",
        "background_color": "#F9F9F9",
        "display":          "standalone",
        "orientation":      "portrait-primary",
        "start_url":        "./",
        "scope":            "./",
        "icons": [
            {
                "src":     "icons/icon-192.png",
                "sizes":   "192x192",
                "type":    "image/png",
                "purpose": "any maskable",
            },
            {
                "src":     "icons/icon-512.png",
                "sizes":   "512x512",
                "type":    "image/png",
                "purpose": "any maskable",
            },
        ],
    }
    with open(DOCS / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("  ✓  manifest.json")


def write_sw():
    sw = r"""// FeedlyVP Service Worker — cache-first with background refresh
const CACHE = 'fvp-v3';
const PRECACHE = ['./','./index.html','./manifest.json','./icons/icon-192.png'];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE)
      .then(c => c.addAll(PRECACHE))
      .catch(() => {})          // don't block install on cache failure
  );
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;
  const url = new URL(e.request.url);
  if (url.origin !== location.origin) return;   // don't intercept external

  e.respondWith(
    caches.open(CACHE).then(cache =>
      cache.match(e.request).then(cached => {
        const network = fetch(e.request).then(res => {
          if (res && res.ok) cache.put(e.request, res.clone());
          return res;
        }).catch(() => cached);                 // offline: return cached
        return cached || network;
      })
    )
  );
});
"""
    (DOCS / "sw.js").write_text(sw, encoding="utf-8")
    print("  ✓  sw.js")


def write_index(data: list):
    if not TEMPLATE.exists():
        print(f"  ✗  Template missing: {TEMPLATE}")
        return

    template = TEMPLATE.read_text(encoding="utf-8")
    data_js  = json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    PLACEHOLDER_OLD = "// __DIGEST_DATA_PLACEHOLDER__\nwindow.DIGEST_DATA = [];"
    PLACEHOLDER_NEW = f"window.DIGEST_DATA = {data_js};"

    if PLACEHOLDER_OLD not in template:
        print("  ⚠  Placeholder not found in template — output may lack data")

    built = template.replace(PLACEHOLDER_OLD, PLACEHOLDER_NEW)

    # Stamp build time in a comment near the top
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    built = built.replace(
        "<!DOCTYPE html>",
        f"<!DOCTYPE html>\n<!-- FeedlyVP PWA — built {stamp} -->",
        1,
    )

    with open(DOCS / "index.html", "w", encoding="utf-8") as f:
        f.write(built)

    total_articles = sum(len(e.get("articles", [])) for e in data)
    print(f"  ✓  index.html  ({len(data)} digests, {total_articles} articles total)")


# ── Main ─────────────────────────────────────────────────────
def main():
    print("\n🔧  Building FeedlyVP PWA…\n")

    data = load_data()
    write_nojekyll()
    write_icons()
    write_manifest()
    write_sw()
    write_index(data)

    file_count = sum(1 for _ in DOCS.rglob("*") if _.is_file())
    print(f"\n✅  Build complete → docs/  ({file_count} files)\n")


if __name__ == "__main__":
    main()
