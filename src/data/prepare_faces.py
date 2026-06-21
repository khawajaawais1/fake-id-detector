"""Download Olivetti faces and save them as JPGs for use in synthetic ID generation."""

from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.datasets import fetch_olivetti_faces


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


def main():
    faces_dir = PROJECT_ROOT / "data" / "raw" / "faces"
    faces_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading Olivetti faces (small, fast)...")
    data = fetch_olivetti_faces()
    images = data.images  # (400, 64, 64) grayscale 0..1

    print(f"Got {len(images)} faces. Upscaling and saving as RGB JPGs...")
    for i, img_arr in enumerate(images):
        arr = (img_arr * 255).astype(np.uint8)
        rgb = np.stack([arr, arr, arr], axis=-1)
        img = Image.fromarray(rgb).resize((200, 250), Image.BICUBIC)
        img.save(faces_dir / f"face_{i:03d}.jpg", quality=90)

    print(f"Saved {len(images)} faces to {faces_dir}")


if __name__ == "__main__":
    main()