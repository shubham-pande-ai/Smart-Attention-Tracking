from pathlib import Path
from PIL import Image

def clean_directory(directory: Path):
    if not directory.exists():
        print(f"Directory not found: {directory}")
        return

    removed = 0
    for file_path in directory.rglob("*"):
        if file_path.is_file():
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Check if it's a valid image
            except Exception as e:
                print(f"Removing corrupted or invalid image: {file_path} - Error: {e}")
                file_path.unlink()
                removed += 1
                
    print(f"Finished cleaning {directory}. Removed {removed} corrupted files.")

if __name__ == "__main__":
    processed_dir = Path("data/processed")
    print(f"Scanning {processed_dir} for corrupted images...")
    clean_directory(processed_dir)
