"""Lightweight video overlay toolkit (PIL): title card, in-frame HUD, head-camera
picture-in-picture, route mini-map, and a final scorecard.

All drawing is pure PIL/numpy so the demo has no heavyweight video dependencies
beyond ``imageio`` + ``pillow``.  Colours are a small dark-tactical palette so the
overlays read clearly over the rendered scene.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_BG = (16, 18, 22)
_FG = (228, 232, 238)
_DIM = (150, 158, 168)
_OK = (90, 210, 130)
_NO = (235, 110, 110)
_ACCENT = (90, 170, 245)
_WARN = (245, 180, 70)


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _bold(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return _font(size)


def _check(d, cx, cy, ok, r=9, pending_dim=True):
    """Draw a status marker as a shape (reliable across fonts): filled green disc with
    a white tick when ``ok``; a hollow ring (or red disc with an x) otherwise."""
    if ok:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_OK)
        d.line([(cx - r * 0.45, cy), (cx - r * 0.1, cy + r * 0.45)], fill=(20, 30, 22), width=2)
        d.line([(cx - r * 0.1, cy + r * 0.45), (cx + r * 0.5, cy - r * 0.5)], fill=(20, 30, 22), width=2)
    elif pending_dim:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=_DIM, width=2)
    else:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_NO)
        d.line([(cx - r * 0.45, cy - r * 0.45), (cx + r * 0.45, cy + r * 0.45)], fill=(40, 20, 20), width=2)
        d.line([(cx - r * 0.45, cy + r * 0.45), (cx + r * 0.45, cy - r * 0.45)], fill=(40, 20, 20), width=2)


def draw_hud(frame: np.ndarray, *, title: str, badge: str, phase: str,
             forward: float, yaw: float, objectives: dict, progress: float,
             route_xy, robot_xy, caption: str = "") -> np.ndarray:
    """Composite the live HUD onto a rendered RGB frame."""
    img = Image.fromarray(frame).convert("RGBA")
    W, H = img.size
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    f_title, f_h, f_b, f_s = _bold(30), _bold(20), _font(18), _font(15)

    # top banner
    d.rectangle([0, 0, W, 52], fill=(*_BG, 200))
    d.text((20, 12), title, font=f_title, fill=_FG)
    bw = d.textlength(badge, font=f_s)
    d.rectangle([W - bw - 34, 14, W - 14, 40], fill=(*_ACCENT, 60))
    d.text((W - bw - 24, 18), badge, font=f_s, fill=_FG)

    # left objectives checklist
    px, py, pw = 18, 70, 300
    rows = list(objectives.items())
    ph = 34 + len(rows) * 26
    d.rectangle([px, py, px + pw, py + ph], fill=(*_BG, 165))
    d.text((px + 14, py + 8), "MISSION OBJECTIVES", font=f_b, fill=_DIM)
    for i, (name, ok) in enumerate(rows):
        yy = py + 36 + i * 26
        _check(d, px + 24, yy + 10, ok, r=8)
        d.text((px + 42, yy + 1), name.replace("_", " "), font=f_b,
               fill=(_FG if ok else _DIM))

    # phase + drive bars (bottom-left)
    by = H - 96
    d.rectangle([18, by, 318, H - 18], fill=(*_BG, 165))
    pcol = _WARN if phase == "SHOVE" else (_ACCENT if phase == "INSPECT" else _OK)
    d.text((34, by + 10), f"STATE: {phase}", font=f_h, fill=pcol)
    _bar(d, 34, by + 44, 270, "drive", forward, _OK)
    _bar(d, 34, by + 66, 270, "turn", (yaw + 1) / 2, _ACCENT)

    # route mini-map (bottom-right)
    _minimap(d, W - 250, H - 120, 232, 100, route_xy, robot_xy, f_s)

    # progress bar (top, under banner)
    d.rectangle([0, 52, int(W * progress), 57], fill=(*_ACCENT, 220))

    if caption:
        cw = d.textlength(caption, font=f_b)
        d.rectangle([(W - cw) / 2 - 14, H - 150, (W + cw) / 2 + 14, H - 120],
                    fill=(*_BG, 180))
        d.text(((W - cw) / 2, H - 146), caption, font=f_b, fill=_FG)

    return np.asarray(Image.alpha_composite(img, ov).convert("RGB"))


def _bar(d, x, y, w, label, frac, col):
    frac = float(np.clip(frac, 0, 1))
    d.text((x, y - 2), label, font=_font(13), fill=_DIM)
    bx = x + 52
    d.rectangle([bx, y, bx + w - 52, y + 12], fill=(60, 64, 70, 200))
    d.rectangle([bx, y, bx + int((w - 52) * frac), y + 12], fill=(*col, 230))


def _minimap(d, x, y, w, h, route_xy, robot_xy, font):
    d.rectangle([x, y, x + w, y + h], fill=(*_BG, 175))
    d.text((x + 10, y + 6), "ROUTE", font=font, fill=_DIM)
    pts = np.asarray(route_xy, dtype=float)
    if len(pts) < 2:
        return
    xs, ys = pts[:, 0], pts[:, 1]
    x0, x1 = xs.min() - 0.4, xs.max() + 0.4
    y0, y1 = ys.min() - 0.4, ys.max() + 0.4
    pad = 14

    def tf(p):
        sx = x + pad + (p[0] - x0) / (x1 - x0) * (w - 2 * pad)
        sy = y + h - pad - (p[1] - y0) / (y1 - y0) * (h - 28)
        return sx, sy
    scr = [tf(p) for p in pts]
    d.line(scr, fill=(*_DIM, 220), width=2)
    for sx, sy in scr:
        d.ellipse([sx - 3, sy - 3, sx + 3, sy + 3], fill=(*_ACCENT, 230))
    rx, ry = tf(robot_xy)
    d.ellipse([rx - 5, ry - 5, rx + 5, ry + 5], fill=(*_OK, 255), outline=_FG)


def draw_pip(frame: np.ndarray, inset: np.ndarray, label: str = "") -> np.ndarray:
    img = Image.fromarray(frame).convert("RGB")
    W, H = img.size
    iw, ih = W // 4, H // 4
    ins = Image.fromarray(inset).convert("RGB").resize((iw, ih))
    px, py = W - iw - 20, 70
    d = ImageDraw.Draw(img)
    d.rectangle([px - 3, py - 3, px + iw + 3, py + ih + 3], fill=_ACCENT)
    img.paste(ins, (px, py))
    if label:
        d.rectangle([px, py + ih - 22, px + iw, py + ih], fill=_BG)
        d.text((px + 8, py + ih - 20), label, font=_font(14), fill=_FG)
    return np.asarray(img)


def _card(w, h):
    img = Image.new("RGB", (w, h), _BG)
    return img, ImageDraw.Draw(img)


def title_card(w, h, *, title, subtitle, bullets, footer) -> np.ndarray:
    img, d = _card(w, h)
    d.text((w // 2 - d.textlength(title, font=_bold(58)) / 2, h * 0.16),
           title, font=_bold(58), fill=_FG)
    d.text((w // 2 - d.textlength(subtitle, font=_font(26)) / 2, h * 0.16 + 78),
           subtitle, font=_font(26), fill=_ACCENT)
    fy = h * 0.40
    for b in bullets:
        d.polygon([(w * 0.16, fy + 4), (w * 0.16, fy + 18), (w * 0.16 + 12, fy + 11)],
                  fill=_ACCENT)
        d.text((w * 0.16 + 24, fy), b, font=_font(23), fill=_FG)
        fy += 46
    d.text((w // 2 - d.textlength(footer, font=_font(19)) / 2, h - 60),
           footer, font=_font(19), fill=_DIM)
    return np.asarray(img)


def scorecard(w, h, *, title, lines, headline, footer) -> np.ndarray:
    img, d = _card(w, h)
    d.text((w // 2 - d.textlength(title, font=_bold(40)) / 2, h * 0.10),
           title, font=_bold(40), fill=_FG)
    d.text((w // 2 - d.textlength(headline, font=_bold(30)) / 2, h * 0.10 + 58),
           headline, font=_bold(30), fill=_OK)
    fy = h * 0.30
    row_h = (h * 0.60) / max(1, len(lines))
    for name, detail, ok in lines:
        _check(d, w * 0.20 + 11, fy + 13, ok, r=10, pending_dim=False)
        d.text((w * 0.20 + 38, fy + 2), name.replace("_", " "), font=_font(23), fill=_FG)
        d.text((w * 0.68, fy + 2), detail, font=_font(20), fill=_DIM)
        fy += row_h
    d.text((w // 2 - d.textlength(footer, font=_font(18)) / 2, h - 52),
           footer, font=_font(18), fill=_DIM)
    return np.asarray(img)
