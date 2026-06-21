"""Generate a synthetic dataset of real and fake ID cards."""

import argparse
import csv
import random
from pathlib import Path

from synthetic_generator import (
    random_fields,
    draw_real_id,
    generate_fake_id,
    add_capture_noise,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


def resolve(path_str: str) -> Path:
    """Resolve a path: if relative, treat it as relative to project root."""
    p = Path(path_str)
    return p if p.is_absolute() else PROJECT_ROOT / p


def main(args):
    out_dir = resolve(args.out_dir)
    real_dir = out_dir / "real"
    fake_dir = out_dir / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    faces_dir = resolve(args.faces_dir)
    face_files = list(faces_dir.glob("*.*")) if faces_dir.exists() else []
    if not face_files:
        raise RuntimeError(
            f"No face images in {faces_dir}.\n"
            f"Run this first:  python src/data/prepare_faces.py"
        )
    print(f"Found {len(face_files)} face images in {faces_dir}")

    random.seed(args.seed)

    manifest = []

    print(f"Generating {args.n_real} real IDs...")
    for i in range(args.n_real):
        fields = random_fields(faces_dir)
        img = draw_real_id(fields)
        img = add_capture_noise(img)
        path = real_dir / f"real_{i:05d}.jpg"
        img.save(path, quality=random.randint(75, 95))
        manifest.append({"path": str(path.relative_to(PROJECT_ROOT)), "label": 0, "tamper": "none", "doc_id": f"real_{i:05d}"})

    print(f"Generating {args.n_fake} fake IDs...")
    for i in range(args.n_fake):
        fields = random_fields(faces_dir)
        img, tamper = generate_fake_id(fields, faces_dir)
        img = add_capture_noise(img)
        path = fake_dir / f"fake_{i:05d}.jpg"
        img.save(path, quality=random.randint(75, 95))
        manifest.append({"path": str(path.relative_to(PROJECT_ROOT)), "label": 1, "tamper": tamper, "doc_id": f"fake_{i:05d}"})

    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "tamper", "doc_id"])
        writer.writeheader()
        writer.writerows(manifest)

    print(f"\nDone. Manifest at {manifest_path}")
    print(f"Real: {args.n_real}  Fake: {args.n_fake}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/raw/synthetic")
    parser.add_argument("--faces_dir", default="data/raw/faces")
    parser.add_argument("--n_real", type=int, default=2000)
    parser.add_argument("--n_fake", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)