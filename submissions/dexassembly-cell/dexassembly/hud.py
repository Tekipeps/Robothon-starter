"""Heads-up-display overlay for demo frames.

Draws a compact telemetry panel (active task, phase, per-finger tactile bars,
total grip force, and an arena progress bar) onto rendered RGB frames using
Pillow, so the demo video clearly communicates what the controller is doing and
that the tactile feedback is live.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FINGERS = ("index", "middle", "ring", "thumb")


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_hud(
    frame: np.ndarray,
    *,
    title: str,
    task: str,
    phase: str,
    forces: np.ndarray,
    grip: float,
    progress: float,
    score_line: str = "",
    caption: str = "",
    badge: str = "",
) -> np.ndarray:
    img = Image.fromarray(frame).convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    f_big = _font(max(18, w // 45))
    f_med = _font(max(14, w // 64))
    f_sm = _font(max(12, w // 80))

    # top banner
    d.rectangle([0, 0, w, h // 11], fill=(12, 14, 18, 200))
    d.text((16, 8), title, font=f_big, fill=(240, 244, 250))
    right_x = w - 16
    if score_line:
        tw = d.textlength(score_line, font=f_med)
        d.text((right_x - tw, 12), score_line, font=f_med, fill=(120, 230, 160))
        right_x -= tw + 16
    # integrity badge: states plainly that the robot is driven by real actuators,
    # not qpos teleportation -- the key honesty claim, kept on-screen throughout.
    if badge:
        bw = d.textlength(badge, font=f_sm)
        pad = 7
        d.rounded_rectangle([right_x - bw - 2 * pad, 9, right_x, 9 + int(f_sm.size) + 2 * pad],
                            radius=6, fill=(20, 60, 36, 220), outline=(90, 200, 120, 255))
        d.text((right_x - bw - pad, 9 + pad), badge, font=f_sm, fill=(150, 235, 175))

    # bottom-left telemetry panel
    px, py, pw, ph = 14, h - h // 4 - 8, w // 3, h // 4
    d.rounded_rectangle([px, py, px + pw, py + ph], radius=10, fill=(12, 14, 18, 175))
    d.text((px + 12, py + 8), f"TASK  {task}", font=f_med, fill=(230, 235, 242))
    d.text((px + 12, py + 8 + (ph // 6)), f"PHASE {phase}", font=f_sm, fill=(170, 200, 230))

    # tactile bars
    bar_x = px + 12
    bar_y = py + ph // 2 - 6
    bar_w = pw - 90
    maxf = max(8.0, float(np.max(forces)) if len(forces) else 1.0)
    for i, name in enumerate(_FINGERS):
        yy = bar_y + i * (ph // 9)
        d.text((bar_x, yy - 1), name[:3].upper(), font=f_sm, fill=(150, 160, 175))
        x0 = bar_x + 42
        frac = float(np.clip(forces[i] / maxf, 0, 1)) if i < len(forces) else 0.0
        d.rectangle([x0, yy, x0 + bar_w, yy + ph // 14], fill=(40, 44, 52, 255))
        col = (90, 200, 120) if frac > 0.02 else (70, 80, 92)
        d.rectangle([x0, yy, x0 + int(bar_w * frac), yy + ph // 14], fill=col)

    d.text((px + 12, py + ph - (ph // 6) - 4),
           f"grip {grip:5.1f} N", font=f_med, fill=(120, 230, 160))

    # narration caption (centered subtitle band) above the progress bar
    if caption:
        words = caption.split()
        lines, cur = [], ""
        max_chars = max(28, w // 12)
        for word in words:
            if len(cur) + len(word) + 1 > max_chars:
                lines.append(cur); cur = word
            else:
                cur = f"{cur} {word}".strip()
        if cur:
            lines.append(cur)
        lh = int(f_med.size * 1.35)
        band_h = lh * len(lines) + 12
        by0 = h - 22 - band_h
        d.rectangle([0, by0, w, by0 + band_h], fill=(8, 10, 14, 180))
        for i, line in enumerate(lines):
            tw = d.textlength(line, font=f_med)
            d.text(((w - tw) / 2, by0 + 6 + i * lh), line, font=f_med, fill=(235, 240, 248))

    # progress bar across the bottom
    pbw = int((w - 28) * float(np.clip(progress, 0, 1)))
    d.rectangle([14, h - 12, w - 14, h - 6], fill=(40, 44, 52, 220))
    d.rectangle([14, h - 12, 14 + pbw, h - 6], fill=(90, 160, 240, 255))
    return np.asarray(img)


def draw_pip(frame: np.ndarray, inset: np.ndarray, label: str = "") -> np.ndarray:
    """Composite a small inset (e.g. the eye-in-hand camera) into the top-right
    corner of ``frame``, with a label — the robot's live point of view."""
    img = Image.fromarray(frame).convert("RGB")
    w, h = img.size
    iw = w // 4
    ih = int(iw * inset.shape[0] / inset.shape[1])
    pip = Image.fromarray(inset).convert("RGB").resize((iw, ih))
    x0, y0 = w - iw - 16, h // 11 + 16
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([x0 - 3, y0 - 3, x0 + iw + 3, y0 + ih + 3], fill=(8, 10, 14, 230),
                outline=(90, 200, 120, 255), width=2)
    img.paste(pip, (x0, y0))
    d = ImageDraw.Draw(img, "RGBA")
    if label:
        f = _font(max(11, w // 90))
        d.rectangle([x0, y0, x0 + iw, y0 + int(f.size) + 8], fill=(8, 10, 14, 180))
        d.text((x0 + 6, y0 + 4), label, font=f, fill=(150, 235, 175))
    return np.asarray(img)


def _card(w: int, h: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (w, h), (10, 12, 16))
    d = ImageDraw.Draw(img, "RGBA")
    # subtle top/bottom accent rules
    d.rectangle([0, 0, w, 6], fill=(90, 160, 240, 255))
    d.rectangle([0, h - 6, w, h], fill=(90, 200, 120, 255))
    return img, d


def title_card(w: int, h: int, *, title: str, subtitle: str,
               bullets: list[str], footer: str = "") -> np.ndarray:
    """A full-frame opening card introducing the cell."""
    img, d = _card(w, h)
    f_title = _font(max(28, w // 24))
    f_sub = _font(max(16, w // 52))
    f_b = _font(max(15, w // 60))
    y = int(h * 0.22)
    d.text((w // 12, y), title, font=f_title, fill=(240, 244, 250))
    y += int(f_title.size * 1.5)
    d.text((w // 12, y), subtitle, font=f_sub, fill=(150, 200, 240))
    y += int(f_sub.size * 2.2)
    for b in bullets:
        d.ellipse([w // 12, y + 6, w // 12 + 9, y + 15], fill=(90, 200, 120, 255))
        d.text((w // 12 + 22, y), b, font=f_b, fill=(215, 222, 230))
        y += int(f_b.size * 1.7)
    if footer:
        fw = d.textlength(footer, font=f_b)
        d.text(((w - fw) / 2, int(h * 0.9)), footer, font=f_b, fill=(120, 130, 145))
    return np.asarray(img)


def scorecard(w: int, h: int, *, title: str, lines: list[tuple[str, str, bool]],
              headline: str, footer: str = "") -> np.ndarray:
    """A full-frame closing card: per-task PASS/FAIL + a headline metric."""
    img, d = _card(w, h)
    f_title = _font(max(26, w // 26))
    f_head = _font(max(20, w // 40))
    f_row = _font(max(15, w // 62))
    d.text((w // 12, int(h * 0.12)), title, font=f_title, fill=(240, 244, 250))
    y = int(h * 0.30)
    for name, detail, ok in lines:
        tag = "PASS" if ok else "FAIL"
        col = (120, 230, 160) if ok else (235, 130, 130)
        d.text((w // 12, y), tag, font=f_row, fill=col)
        d.text((w // 12 + int(w * 0.07), y), name, font=f_row, fill=(225, 230, 238))
        d.text((w // 12 + int(w * 0.40), y), detail, font=f_row, fill=(150, 160, 175))
        y += int(f_row.size * 1.7)
    y += int(f_row.size * 0.6)
    hw = d.textlength(headline, font=f_head)
    d.text(((w - hw) / 2, y), headline, font=f_head, fill=(120, 230, 160))
    if footer:
        fw = d.textlength(footer, font=f_row)
        d.text(((w - fw) / 2, int(h * 0.9)), footer, font=f_row, fill=(120, 130, 145))
    return np.asarray(img)
