import argparse
import random
import re
import shutil
from pathlib import Path

from PIL import Image


COORD_PATTERN = re.compile(r"(frame_\d+\.jpg):\s*([-0-9.]+),\s*([-0-9.]+)")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def reset_dir(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            path.unlink()


def read_center(participant_dir: Path) -> tuple[float, float] | None:
    center_file = participant_dir / "C" / "C.txt"
    if not center_file.exists():
        return None
    text = center_file.read_text(encoding="utf-8", errors="ignore").strip()
    match = re.search(r"([-0-9.]+)\s*,\s*([-0-9.]+)", text)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def read_coordinates(path: Path) -> list[tuple[Path, float, float]]:
    records = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = COORD_PATTERN.search(line)
        if not match:
            continue
        image_path = path.parent / match.group(1)
        if image_path.exists():
            records.append((image_path, float(match.group(2)), float(match.group(3))))
    return records


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def distance_from_center(x: float, y: float, center_x: float, center_y: float, width: int, height: int) -> float:
    dx = (x - center_x) / width
    dy = (y - center_y) / height
    return (dx * dx + dy * dy) ** 0.5


def collect_gaze_samples(
    gazetrack_root: Path,
    screen_radius: float,
    away_radius: float,
) -> tuple[list[Path], list[Path]]:
    screen_samples: list[Path] = []
    away_samples: list[Path] = []

    for coord_file in gazetrack_root.rglob("coordinates_sorted.txt"):
        records = read_coordinates(coord_file)
        if not records:
            continue

        participant_dir = coord_file.parents[2]
        center = read_center(participant_dir)
        if center is None:
            xs = sorted(record[1] for record in records)
            ys = sorted(record[2] for record in records)
            center = xs[len(xs) // 2], ys[len(ys) // 2]

        width, height = image_size(records[0][0])
        for image_path, x, y in records:
            distance = distance_from_center(x, y, center[0], center[1], width, height)
            if distance <= screen_radius:
                screen_samples.append(image_path)
            elif distance >= away_radius:
                away_samples.append(image_path)

    return screen_samples, away_samples


def copy_samples(samples: list[Path], target: Path, prefix: str, max_count: int) -> int:
    target.mkdir(parents=True, exist_ok=True)
    count = min(len(samples), max_count)
    for index, source in enumerate(samples[:count]):
        target_path = target / f"{prefix}_{index:06d}{source.suffix.lower()}"
        shutil.copy2(source, target_path)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert GazeTrack coordinate data into attention source folders.")
    parser.add_argument("--gazetrack-root", type=Path, default=Path("data/raw/gazetrack/GazeTrack"))
    parser.add_argument("--output", type=Path, default=Path("data/source"))
    parser.add_argument("--max-per-class", type=int, default=6000)
    parser.add_argument("--screen-radius", type=float, default=0.08)
    parser.add_argument("--away-radius", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean-gaze", action="store_true", help="Clear only looking_screen and looking_away first.")
    args = parser.parse_args()

    if not args.gazetrack_root.exists():
        raise FileNotFoundError(f"GazeTrack root not found: {args.gazetrack_root}")

    screen_dir = args.output / "looking_screen"
    away_dir = args.output / "looking_away"
    if args.clean_gaze:
        reset_dir(screen_dir)
        reset_dir(away_dir)

    screen_samples, away_samples = collect_gaze_samples(args.gazetrack_root, args.screen_radius, args.away_radius)
    random.seed(args.seed)
    random.shuffle(screen_samples)
    random.shuffle(away_samples)

    screen_count = copy_samples(screen_samples, screen_dir, "gazetrack_screen", args.max_per_class)
    away_count = copy_samples(away_samples, away_dir, "gazetrack_away", args.max_per_class)

    print(f"GazeTrack candidates: screen={len(screen_samples)}, away={len(away_samples)}")
    print(f"Copied: screen={screen_count}, away={away_count}")
    print("Eyes-closed data should still come from MRL or your own webcam calibration data.")


if __name__ == "__main__":
    main()
