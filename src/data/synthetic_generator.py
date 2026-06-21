"""Synthetic ID card generator — cross-platform (Windows / Linux / macOS)."""

import random
import string
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


CARD_SIZE = (640, 400)
PHOTO_BOX = (40, 110, 200, 310)

FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf",
    "C:/Windows/Fonts/verdana.ttf", "C:/Windows/Fonts/tahoma.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/Library/Fonts/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc",
]

WRONG_FONTS = [
    "C:/Windows/Fonts/comic.ttf", "C:/Windows/Fonts/impact.ttf", "C:/Windows/Fonts/times.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/Library/Fonts/Times New Roman.ttf",
]

MONO_FONTS = [
    "C:/Windows/Fonts/cour.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
]


def _available_fonts(candidates):
    return [p for p in candidates if Path(p).exists()]


_REAL_FONTS = _available_fonts(FONT_CANDIDATES)
_WRONG_FONTS_AVAILABLE = _available_fonts(WRONG_FONTS) or _REAL_FONTS
_MONO_FONTS_AVAILABLE = _available_fonts(MONO_FONTS) or _REAL_FONTS

if not _REAL_FONTS:
    raise RuntimeError("No usable real fonts found.")

print(f"[synthetic_generator] real={len(_REAL_FONTS)} wrong={len(_WRONG_FONTS_AVAILABLE)} mono={len(_MONO_FONTS_AVAILABLE)}")


def _pick_real_font(size): return ImageFont.truetype(random.choice(_REAL_FONTS), size)
def _pick_wrong_font(size): return ImageFont.truetype(random.choice(_WRONG_FONTS_AVAILABLE), size)
def _pick_mono_font(size): return ImageFont.truetype(random.choice(_MONO_FONTS_AVAILABLE), size)


FIRST_NAMES = ["Alex", "Maria", "John", "Aino", "Sami", "Linnea", "Mikko", "Sara", "Erik", "Anna"]
LAST_NAMES = ["Virtanen", "Korhonen", "Niemi", "Mäkinen", "Heikkinen", "Laine", "Salonen"]
COUNTRIES = ["FINLAND", "ESTONIA", "SWEDEN", "NORWAY", "DENMARK"]


@dataclass
class IDFields:
    name: str
    dob: str
    id_number: str
    expiry: str
    country: str
    photo_path: str


def random_fields(faces_dir: Path) -> IDFields:
    return IDFields(
        name=f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
        dob=f"{random.randint(1, 28):02d}/{random.randint(1, 12):02d}/{random.randint(1960, 2005)}",
        id_number="".join(random.choices(string.digits, k=9)),
        expiry=f"{random.randint(1, 28):02d}/{random.randint(1, 12):02d}/{random.randint(2026, 2035)}",
        country=random.choice(COUNTRIES),
        photo_path=str(random.choice(list(faces_dir.glob("*.*")))),
    )


def draw_real_id(fields: IDFields) -> Image.Image:
    img = Image.new("RGB", CARD_SIZE, color=(245, 245, 240))
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, CARD_SIZE[0], 70], fill=(20, 60, 130))
    title_font = _pick_real_font(28)
    draw.text((30, 20), f"{fields.country} - ID CARD", font=title_font, fill="white")

    photo = Image.open(fields.photo_path).convert("RGB")
    photo = photo.resize((PHOTO_BOX[2] - PHOTO_BOX[0], PHOTO_BOX[3] - PHOTO_BOX[1]))
    img.paste(photo, (PHOTO_BOX[0], PHOTO_BOX[1]))
    draw.rectangle(PHOTO_BOX, outline=(0, 0, 0), width=2)

    label_font = _pick_real_font(16)
    value_font = _pick_real_font(20)

    rows = [
        ("Name", fields.name, 110),
        ("Date of Birth", fields.dob, 170),
        ("ID Number", fields.id_number, 230),
        ("Expiry", fields.expiry, 290),
    ]
    for label, value, y in rows:
        draw.text((230, y), label, font=label_font, fill=(80, 80, 80))
        draw.text((230, y + 20), value, font=value_font, fill=(0, 0, 0))

    for i in range(0, CARD_SIZE[0], 8):
        draw.line([(i, 350), (i + 20, 380)], fill=(180, 200, 220), width=1)

    holo = Image.new("RGBA", (120, 50), (200, 180, 255, 80))
    img.paste(holo, (500, 130), holo)

    mrz_font = _pick_mono_font(14)
    mrz_text = "P<" + fields.country[:3] + "<<" + fields.name.replace(" ", "<").upper()[:30]
    mrz_text2 = fields.id_number + "<<<<<<<<<<<<" + fields.dob.replace("/", "")
    draw.text((20, 350), mrz_text[:40], font=mrz_font, fill=(0, 0, 0))
    draw.text((20, 370), mrz_text2[:40], font=mrz_font, fill=(0, 0, 0))

    return img


def tamper_photo_swap(img: Image.Image, faces_dir: Path) -> Image.Image:
    out = img.copy()
    new_photo = Image.open(random.choice(list(faces_dir.glob("*.*")))).convert("RGB")
    new_photo = new_photo.resize((PHOTO_BOX[2] - PHOTO_BOX[0], PHOTO_BOX[3] - PHOTO_BOX[1]))
    offset_x = random.randint(-5, 5)
    offset_y = random.randint(-5, 5)
    out.paste(new_photo, (PHOTO_BOX[0] + offset_x, PHOTO_BOX[1] + offset_y))
    return out


def tamper_wrong_font(img: Image.Image, fields: IDFields) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle([230, 168, 480, 200], fill=(245, 245, 240))
    wrong_font = _pick_wrong_font(20)
    draw.text((230, 170), fields.name, font=wrong_font, fill=(0, 0, 0))
    return out


def tamper_text_edit(img: Image.Image, fields: IDFields) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle([230, 188, 400, 220], fill=(245, 245, 240))
    altered_dob = f"{random.randint(1, 28):02d}/{random.randint(1, 12):02d}/{random.randint(1970, 1990)}"
    font = _pick_real_font(20)
    draw.text((230, 190), altered_dob, font=font, fill=(0, 0, 0))
    return out


def tamper_color_shift(img: Image.Image) -> Image.Image:
    arr = np.array(img).astype(np.int16)
    shift = np.array([random.randint(-25, 25), random.randint(-15, 15), random.randint(-15, 15)])
    arr = np.clip(arr + shift, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def tamper_remove_hologram(img: Image.Image) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle([500, 130, 620, 180], fill=(245, 245, 240))
    return out


def tamper_copy_artifacts(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=0.8))


TAMPERING_OPS = [
    ("photo_swap", tamper_photo_swap),
    ("wrong_font", tamper_wrong_font),
    ("text_edit", tamper_text_edit),
    ("color_shift", lambda img, *_: tamper_color_shift(img)),
    ("remove_hologram", lambda img, *_: tamper_remove_hologram(img)),
    ("copy_artifacts", lambda img, *_: tamper_copy_artifacts(img)),
]


def generate_fake_id(fields: IDFields, faces_dir: Path) -> Tuple[Image.Image, str]:
    base = draw_real_id(fields)
    n_tampers = random.randint(1, 3)
    chosen = random.sample(TAMPERING_OPS, n_tampers)
    out = base
    tamper_types = []
    for name, op in chosen:
        if name == "photo_swap":
            out = op(out, faces_dir)
        elif name in ("wrong_font", "text_edit"):
            out = op(out, fields)
        else:
            out = op(out)
        tamper_types.append(name)
    return out, "+".join(tamper_types)


def add_capture_noise(img: Image.Image) -> Image.Image:
    angle = random.uniform(-5, 5)
    img = img.rotate(angle, fillcolor=(255, 255, 255), resample=Image.BICUBIC)
    arr = np.array(img).astype(np.float32)
    brightness = random.uniform(0.85, 1.15)
    arr = np.clip(arr * brightness, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)