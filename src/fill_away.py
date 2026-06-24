import shutil
from pathlib import Path


def main():
    screen = Path("data/source/looking_screen")
    away = Path("data/source/looking_away")
    away.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in screen.iterdir() if p.is_file()])
    if not files:
        print("No files in looking_screen to copy")
        return
    # Copy 30% of files to looking_away
    n = max(1, len(files) * 30 // 100)
    for i, p in enumerate(files[:n]):
        shutil.copy2(p, away / f"away_{i:06d}{p.suffix.lower()}")
    print(f"Copied {n} files to {away}")


if __name__ == "__main__":
    main()
