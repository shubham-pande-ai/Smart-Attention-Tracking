import argparse
import time
from pathlib import Path

import cv2


CLASS_NAMES = {"looking_screen", "looking_away", "eyes_closed"}


def crop_eye_region(
    frame,
    face_cascade: cv2.CascadeClassifier,
    eye_cascade: cv2.CascadeClassifier,
):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None, None

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
    else:
        x1 = x + int(w * 0.18)
        y1 = y + int(h * 0.18)
        x2 = x + int(w * 0.82)
        y2 = y + int(h * 0.45)

    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect labeled webcam eye crops for attention training.")
    parser.add_argument("--class-name", required=True, choices=sorted(CLASS_NAMES))
    parser.add_argument("--output", type=Path, default=Path("data/source"))
    parser.add_argument("--count", type=int, default=600)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.12)
    args = parser.parse_args()

    target_dir = args.output / args.class_name
    target_dir.mkdir(parents=True, exist_ok=True)

    existing = len([path for path in target_dir.iterdir() if path.is_file()])
    saved = 0
    collecting = False
    last_saved_at = 0.0

    capture = cv2.VideoCapture(args.camera_index)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    print(f"Collecting class: {args.class_name}")
    print("Press S to start/pause. Press Q to quit.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Could not read from webcam.")

            frame = cv2.flip(frame, 1)
            crop, box = crop_eye_region(frame, face_cascade, eye_cascade)
            status = "COLLECTING" if collecting else "PAUSED"
            color = (45, 190, 90) if collecting else (0, 165, 255)

            if box is not None:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            now = time.time()
            if collecting and crop is not None and crop.size > 0 and now - last_saved_at >= args.interval:
                filename = target_dir / f"{args.class_name}_{existing + saved:06d}.png"
                cv2.imwrite(str(filename), crop)
                saved += 1
                last_saved_at = now
                if saved >= args.count:
                    break

            cv2.putText(frame, f"{args.class_name} | {status} | {saved}/{args.count}", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, "S=start/pause  Q=quit", (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Collect Calibration Data", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                collecting = not collecting
            elif key == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print(f"Saved {saved} images to {target_dir}")


if __name__ == "__main__":
    main()
