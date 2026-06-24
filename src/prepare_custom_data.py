import argparse
import random
import shutil
import csv
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def collect_images(folder: Path) -> list[Path]:
    return [path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]

def reset_dir(folder: Path) -> None:
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True, exist_ok=True)

def verify_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except:
        return False

def load_daisee_labels(labels_dir: Path) -> dict:
    mapping = {}
    for split, filename in [("train", "TrainLabels.csv"), ("val", "ValidationLabels.csv"), ("test", "TestLabels.csv")]:
        csv_path = labels_dir / filename
        if not csv_path.exists():
            continue
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 5:
                    clip_id = row[0].strip()
                    mapping[clip_id] = {
                        "boredom": int(row[1]),
                        "engagement": int(row[2]),
                        "confusion": int(row[3]),
                        "frustration": int(row[4]),
                        "split": split
                    }
    return mapping

def extract_dir_images(source_dir: Path, output_dir: Path, state_str: str) -> list[dict]:
    if not source_dir.exists():
        print(f"Directory not found: {source_dir}")
        return []

    print(f"Extracting valid images from {source_dir}...")
    images = collect_images(source_dir)
    valid_records = []
    
    for idx, img_path in enumerate(images):
        if verify_image(img_path):
            filename = f"oace_{state_str}_{idx:06d}{img_path.suffix.lower()}"
            target = output_dir / filename
            shutil.copy2(img_path, target)
            valid_records.append({
                "image": filename,
                "eye_state": 1 if state_str == "open" else 0,
                "boredom": -1,
                "engagement": -1,
                "confusion": -1,
                "frustration": -1,
                "src_split": "random"
            })
            
    return valid_records

def process_daisee_videos(daisee_root: Path, labels_dir: Path, output_dir: Path, max_videos: int) -> list[dict]:
    if not daisee_root.exists():
        print(f"DAiSEE root not found: {daisee_root}")
        return []
        
    mapping = load_daisee_labels(labels_dir)
    videos = []
    for ext in ["*.avi", "*.mp4"]:
        videos.extend(list(daisee_root.rglob(ext)))
        
    if not videos:
        return []
        
    random.shuffle(videos)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    
    records = []
    video_count = 0
    saved_count = 0
    
    for vid_path in videos:
        if video_count >= max_videos:
            break
            
        clip_id = vid_path.name
        if clip_id not in mapping:
            clip_id_avi = vid_path.stem + ".avi"
            clip_id_mp4 = vid_path.stem + ".mp4"
            if clip_id_avi in mapping:
                clip_id = clip_id_avi
            elif clip_id_mp4 in mapping:
                clip_id = clip_id_mp4
            else:
                continue
                
        labels = mapping[clip_id]
        cap = cv2.VideoCapture(str(vid_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % fps == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
                    upper_face_gray = gray[y : y + h // 2, x : x + w]
                    eyes = eye_cascade.detectMultiScale(upper_face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
                    if len(eyes) > 0:
                        ex, ey, ew, eh = max(eyes, key=lambda box: box[2] * box[3])
                        pad = 12
                        x1 = max(x + ex - pad, 0)
                        y1 = max(y + ey - pad, 0)
                        x2 = min(x + ex + ew + pad, frame.shape[1])
                        y2 = min(y + ey + eh + pad, frame.shape[0])
                        
                        eye_crop = frame[y1:y2, x1:x2]
                        if eye_crop.size > 0:
                            filename = f"daisee_{saved_count:06d}.jpg"
                            target = output_dir / filename
                            cv2.imwrite(str(target), eye_crop)
                            records.append({
                                "image": filename,
                                "eye_state": 1,
                                "boredom": labels["boredom"],
                                "engagement": labels["engagement"],
                                "confusion": labels["confusion"],
                                "frustration": labels["frustration"],
                                "src_split": labels["split"]
                            })
                            saved_count += 1
                            
            frame_count += 1
        cap.release()
        video_count += 1
        
    print(f"Extracted {saved_count} DAiSEE frames from {video_count} videos.")
    return records

def assign_splits(records: list[dict], train_ratio: float, val_ratio: float):
    # Some DAiSEE records have predetermined splits. Zip records are "random"
    random_records = [r for r in records if r["src_split"] == "random"]
    random.shuffle(random_records)
    
    total = len(random_records)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    for i, r in enumerate(random_records):
        if i < train_end:
            r["split"] = "train"
        elif i < val_end:
            r["split"] = "val"
        else:
            r["split"] = "test"
            
    for r in records:
        if r["src_split"] != "random":
            r["split"] = r["src_split"]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oace-root", type=Path, default=Path(r"C:\Users\HP\OneDrive\Desktop\archive (1)\OACE"))
    parser.add_argument("--daisee-root", type=Path, default=Path(r"D:\AI\Project\Dataset\DAiSEE\DAiSEE\DataSet"))
    parser.add_argument("--daisee-labels", type=Path, default=Path(r"D:\AI\Project\Dataset\DAiSEE\DAiSEE\Labels"))
    parser.add_argument("--processed-out", type=Path, default=Path("data/processed"))
    parser.add_argument("--max-daisee-videos", type=int, default=500)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    
    print("Cleaning processed directory...")
    reset_dir(args.processed_out)
    images_dir = args.processed_out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Collect all images
    closed_records = extract_dir_images(args.oace_root / "close", images_dir, "closed")
    open_oace_records = extract_dir_images(args.oace_root / "open", images_dir, "open")
    open_daisee_records = process_daisee_videos(args.daisee_root, args.daisee_labels, images_dir, args.max_daisee_videos)
    
    # 2. Balance classes: eyes_open must equal eyes_closed perfectly
    total_closed = len(closed_records)
    print(f"\nInitial valid counts -> Closed: {total_closed} | Open (OACE): {len(open_oace_records)} | Open (DAiSEE): {len(open_daisee_records)}")
    
    # We want len(open_final) == len(closed_final)
    # Prioritize DAiSEE to ensure we learn emotions
    open_final = list(open_daisee_records)
    
    if len(open_final) < total_closed:
        # We need more open eyes, take from OACE
        needed = total_closed - len(open_final)
        random.shuffle(open_oace_records)
        open_final.extend(open_oace_records[:needed])
    elif len(open_final) > total_closed:
        # We have too many DAiSEE images, randomly subsample them down to match closed
        random.shuffle(open_final)
        open_final = open_final[:total_closed]
        
    # We now guarantee len(open_final) <= len(closed_final). Trim closed_final to match if needed.
    final_target = min(len(closed_records), len(open_final))
    
    random.shuffle(closed_records)
    closed_final = closed_records[:final_target]
    open_final = open_final[:final_target]
    
    print(f"BALANCED DATASET -> Closed: {len(closed_final)} | Open: {len(open_final)}")
    
    # 3. Apply Train/Val/Test Splits
    all_records = closed_final + open_final
    assign_splits(all_records, args.train_ratio, args.val_ratio)
    
    # 4. Write CSV
    csv_path = args.processed_out / "dataset.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["split", "image", "eye_state", "boredom", "engagement", "confusion", "frustration"])
        for r in all_records:
            writer.writerow([r["split"], r["image"], r["eye_state"], r["boredom"], r["engagement"], r["confusion"], r["frustration"]])
            
    print(f"\nBalanced Multi-task dataset saved to {args.processed_out}")

if __name__ == "__main__":
    main()
