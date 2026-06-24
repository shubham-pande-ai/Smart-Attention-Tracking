import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(folder: Path) -> list[Path]:
    return [path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]


def reset_dir(folder: Path) -> None:
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True, exist_ok=True)


def copy_split(images: list[Path], class_name: str, output: Path, train_ratio: float, val_ratio: float) -> None:
    random.shuffle(images)
    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }

    for split, split_images in splits.items():
        target_dir = output / split / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for index, source in enumerate(split_images):
            target = target_dir / f"{class_name}_{index:06d}{source.suffix.lower()}"
            shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/val/test folders for attention CNN.")
    parser.add_argument("--looking-screen", required=True, type=Path)
    parser.add_argument("--looking-away", required=True, type=Path)
    parser.add_argument("--eyes-closed", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-ratio", default=0.7, type=float)
    parser.add_argument("--val-ratio", default=0.15, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--clean", action="store_true", help="Delete existing processed output before copying.")
    args = parser.parse_args()

    random.seed(args.seed)
    if args.clean:
        reset_dir(args.output)

    sources = {
        "looking_screen": args.looking_screen,
        "looking_away": args.looking_away,
        "eyes_closed": args.eyes_closed,
    }

    for class_name, folder in sources.items():
        if not folder.exists():
            raise FileNotFoundError(f"Missing source folder: {folder}")
        images = collect_images(folder)
        if not images:
            raise RuntimeError(f"No images found for {class_name} in {folder}")
        copy_split(images, class_name, args.output, args.train_ratio, args.val_ratio)
        print(f"{class_name}: copied {len(images)} images")

    print(f"Processed dataset written to {args.output}")


if __name__ == "__main__":
    main()
