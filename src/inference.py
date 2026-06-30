import cv2
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

IMG_SIZE = (160, 160)
CLASS_NAMES = ["eyes_closed", "eyes_open"]

@dataclass
class Prediction:
    label: str
    confidence: float
    face_detected: bool
    box: tuple[int, int, int, int] | None = None
    emotions: dict[str, float] = field(default_factory=dict)

class AttentionEngine:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.model = None
        self.model_error = ""
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        self.load_model()

    def load_model(self) -> None:
        if not self.model_path.exists():
            self.model_error = f"Model not found: {self.model_path}"
            return
        try:
            self.model = torch.jit.load(str(self.model_path))
            self.model.eval()
            self.model_error = ""
        except Exception as exc:
            self.model_error = str(exc)

    def crop_eye_region(self, frame: np.ndarray) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            return None, None

        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        upper_face_gray = gray[y : y + h // 2, x : x + w]
        eyes = self.eye_cascade.detectMultiScale(upper_face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
        
        if len(eyes) > 0:
            ex, ey, ew, eh = max(eyes, key=lambda box: box[2] * box[3])
            pad = 12
            x1 = max(x + ex - pad, 0)
            y1 = max(y + ey - pad, 0)
            x2 = min(x + ex + ew + pad, frame.shape[1])
            y2 = min(y + ey + eh + pad, frame.shape[0])
        else:
            ew = int(w * 0.25)
            eh = int(h * 0.20)
            ex = int(w * 0.25)
            ey = int(h * 0.25)
            pad = 12
            x1 = max(x + ex - pad, 0)
            y1 = max(y + ey - pad, 0)
            x2 = min(x + ex + ew + pad, frame.shape[1])
            y2 = min(y + ey + eh + pad, frame.shape[0])
            
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

    def predict(self, frame: np.ndarray) -> Prediction:
        if self.model is None:
            return Prediction("model_unavailable", 0.0, False, None)
        eye_crop, box = self.crop_eye_region(frame)
        if eye_crop is None or eye_crop.size == 0:
            return Prediction("no_face", 0.0, False, None)

        rgb = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, IMG_SIZE)
        
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        batch = tensor.unsqueeze(0)
        
        device = next(self.model.parameters()).device if hasattr(self.model, "parameters") else torch.device("cpu")
        batch = batch.to(device)
        
        with torch.no_grad():
            outputs = self.model(batch)
            probs_eye = torch.nn.functional.softmax(outputs["eye_state"][0], dim=0).cpu().numpy()
            
        index = int(np.argmax(probs_eye))
        confidence = float(probs_eye[index])
        label = CLASS_NAMES[index]
        
        emotions = {}
        for em in ["boredom", "engagement", "confusion", "frustration"]:
            probs_em = torch.nn.functional.softmax(outputs[em][0], dim=0).cpu().numpy()
            # The model outputs a class from 0 to 3. Scale it to 0-100% for the dashboard!
            raw_class = float(np.argmax(probs_em))
            emotions[em] = round(raw_class * (100.0 / 3.0), 2)
            
        if label == "eyes_closed":
            emotions["boredom"] = 100.0
            emotions["engagement"] = 0.0
            emotions["confusion"] = 0.0
            emotions["frustration"] = 0.0
            
        return Prediction(label, confidence, True, box, emotions=emotions)
