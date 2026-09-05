"""Generate the app icons from one description, so they cannot drift apart.

    python tools/make_icons.py

Writes PNGs into tools/webui/. Run it again after changing the brand colours;
the results are committed, so the kit itself never needs Pillow at runtime.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent / "webui"
A = (108, 124, 255)      # --accent, dark theme
B = (167, 139, 250)      # --accent-2
SS = 4                   # supersample factor: draw big, shrink, get clean edges


def gradient(size: int) -> Image.Image:
    img = Image.new("RGB", (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            t = (x + y) / (2 * (size - 1))
            px[x, y] = (round(A[0] + (B[0] - A[0]) * t),
                        round(A[1] + (B[1] - A[1]) * t),
                        round(A[2] + (B[2] - A[2]) * t))
    return img


def icon(size: int, *, radius_ratio: float = 0.235, dot_ratio: float = 0.30,
         pad_ratio: float = 0.0) -> Image.Image:
    """Rounded-square mark with a white dot - the brand mark from the UI.
    `pad_ratio` insets the art for maskable icons, whose outer 10% may be
    cropped to whatever shape the launcher prefers."""
    s = size * SS
    pad = round(s * pad_ratio)
    box = s - 2 * pad

    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [pad, pad, pad + box - 1, pad + box - 1], radius=round(box * radius_ratio), fill=255)

    art = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    art.paste(gradient(s), (0, 0), mask)

    d = ImageDraw.Draw(art)
    r = box * dot_ratio / 2
    cx = cy = pad + box / 2
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255, 255))
    return art.resize((size, size), Image.LANCZOS)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("icon-192.png", 192, {}),
        ("icon-512.png", 512, {}),
        # maskable: art inside the 80% safe zone, background out to the edges
        ("icon-maskable-512.png", 512, {"radius_ratio": 0.5, "dot_ratio": 0.34, "pad_ratio": 0.0}),
        ("apple-touch-icon.png", 180, {"radius_ratio": 0.0}),
    ]
    for name, size, kw in jobs:
        img = icon(size, **kw)
        if name == "icon-maskable-512.png":
            # a full-bleed square: launchers crop it to their own shape
            bg = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            bg.paste(gradient(size), (0, 0))
            d = ImageDraw.Draw(bg)
            r = size * 0.17
            d.ellipse([size / 2 - r, size / 2 - r, size / 2 + r, size / 2 + r],
                      fill=(255, 255, 255, 255))
            img = bg
        img.save(OUT / name, optimize=True)
        print(f"  {name}  {size}x{size}  {(OUT / name).stat().st_size} bytes")

    # Windows wants one .ico carrying every size it might draw: 16 px in the
    # tray, 32 in the taskbar, 256 for the large icon view in Explorer.
    ico = Path(__file__).resolve().parent / "simplex.ico"
    icon(256).save(ico, sizes=[(16, 16), (20, 20), (24, 24), (32, 32),
                               (48, 48), (64, 64), (128, 128), (256, 256)])
    print(f"  {ico.name}  multi-size  {ico.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
