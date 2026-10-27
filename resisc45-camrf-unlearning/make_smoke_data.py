"""Create a tiny synthetic ImageFolder dataset for plumbing tests only."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def create_dataset(root: Path, classes: int = 4, images_per_class: int = 30) -> None:
    rng = np.random.default_rng(57)
    for class_id in range(classes):
        folder = root / f"class_{class_id}"
        folder.mkdir(parents=True, exist_ok=True)
        for image_id in range(images_per_class):
            image = rng.integers(0, 60, (32, 32, 3), dtype=np.uint8)
            image[:, :, class_id % 3] += 140
            image[class_id * 4 : class_id * 4 + 5, :] = 255
            Image.fromarray(image).save(folder / f"{image_id:03d}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("smoke_data"))
    args = parser.parse_args()
    create_dataset(args.output)
    print(f"Synthetic data saved to {args.output.resolve()} (not experiment data).")


if __name__ == "__main__":
    main()
