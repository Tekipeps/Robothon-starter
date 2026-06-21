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
    if score_line:
        tw = d.textlength(score_line, font=f_med)
        d.text((w - tw - 16, 12), score_line, font=f_med, fill=(120, 230, 160))

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
