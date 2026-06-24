import argparse
import shutil
import random
from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect(folder: Path):
    if not folder.exists():
        return []
    return [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXT]


def reset_dir(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for path in sorted(folder.rglob("*"), reverse=True):
        try:
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        except OSError as exc:
            print(f"Warning: could not remove {path}: {exc}")


def copy_images(images: list[Path], target_dir: Path, prefix: str) -> None:
    for i, source in enumerate(images):
        target = target_dir / f"{prefix}_{i:06d}{source.suffix.lower()}"
        shutil.copy2(source, target)


def main():
    parser = argparse.ArgumentParser(description="Build source folders from downloaded eye datasets.")
    parser.add_argument("--max-per-class", type=int, default=3000)
    args = parser.parse_args()

    root = Path("data/raw")
    # MRL dataset locations
    mrl1 = root / "mrl-eye-dataset" / "data"
    mrl2 = root / "mrl-eye-dataset-full" / "data"

    awake_sources = []
    sleepy_sources = []
    for base in (mrl1, mrl2):
        awake_sources.extend(collect(base / "train" / "awake"))
        awake_sources.extend(collect(base / "test" / "awake"))
        sleepy_sources.extend(collect(base / "train" / "sleepy"))
        sleepy_sources.extend(collect(base / "test" / "sleepy"))

    awake_sources = list(dict.fromkeys(awake_sources))
    sleepy_sources = list(dict.fromkeys(sleepy_sources))

    random.seed(42)
    random.shuffle(awake_sources)
    random.shuffle(sleepy_sources)

    if len(awake_sources) < 2 or not sleepy_sources:
        raise RuntimeError(
            "Not enough MRL images found. Expected awake and sleepy images under data/raw/mrl-eye-dataset/data."
        )

    out_base = Path("data/source")
    screen_dir = out_base / "looking_screen"
    away_dir = out_base / "looking_away"
    closed_dir = out_base / "eyes_closed"

    for d in (screen_dir, away_dir, closed_dir):
        reset_dir(d)

    max_per_class = args.max_per_class
    screen_images = awake_sources[:max_per_class]
    away_images = awake_sources[max_per_class : max_per_class * 2]
    closed_images = sleepy_sources[:max_per_class]

    if len(away_images) < max_per_class:
        raise RuntimeError("Not enough awake images to create both looking_screen and looking_away demo classes.")

    copy_images(screen_images, screen_dir, "screen")
    copy_images(away_images, away_dir, "away")
    copy_images(closed_images, closed_dir, "closed")

    print(
        "Prepared sources: "
        f"screen={len(list(screen_dir.iterdir()))}, "
        f"away={len(list(away_dir.iterdir()))}, "
        f"closed={len(list(closed_dir.iterdir()))}"
    )
    print("Note: screen/away are demo labels from awake-eye images. Use MPII/GazeTrack labels for true gaze direction.")


if __name__ == "__main__":
    main()
