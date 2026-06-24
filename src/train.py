import argparse
import time
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from tqdm import tqdm
from PIL import Image

IMG_SIZE = (160, 160)

class MultiTaskDataset(Dataset):
    def __init__(self, csv_file: Path, img_dir: Path, split: str, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = []
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if row[0] == split:
                    self.data.append({
                        "image": row[1],
                        "eye_state": int(row[2]),
                        "boredom": int(row[3]),
                        "engagement": int(row[4]),
                        "confusion": int(row[5]),
                        "frustration": int(row[6]),
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = self.img_dir / item["image"]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        labels = {
            "eye_state": torch.tensor(item["eye_state"], dtype=torch.long),
            "boredom": torch.tensor(item["boredom"], dtype=torch.long),
            "engagement": torch.tensor(item["engagement"], dtype=torch.long),
            "confusion": torch.tensor(item["confusion"], dtype=torch.long),
            "frustration": torch.tensor(item["frustration"], dtype=torch.long),
        }
        return image, labels

class MultiHeadEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        self.features = base.features
        
        # FINE TUNING: Freeze all early layers to prevent overfitting on our small dataset
        for param in self.features.parameters():
            param.requires_grad = False
            
        # Unfreeze the last 3 blocks of the network to allow it to adapt to eye/emotion features
        for param in self.features[-3:].parameters():
            param.requires_grad = True
            
        self.pool = base.avgpool
        in_features = base.classifier[1].in_features
        
        self.eye_state = nn.Linear(in_features, 2)
        self.boredom = nn.Linear(in_features, 4)
        self.engagement = nn.Linear(in_features, 4)
        self.confusion = nn.Linear(in_features, 4)
        self.frustration = nn.Linear(in_features, 4)
        
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return {
            "eye_state": self.eye_state(x),
            "boredom": self.boredom(x),
            "engagement": self.engagement(x),
            "confusion": self.confusion(x),
            "frustration": self.frustration(x)
        }

def compute_loss(outputs, labels, criterion):
    # Eye state is heavily weighted as it is the primary task
    loss = 2.0 * criterion(outputs["eye_state"], labels["eye_state"])
    
    mask = labels["boredom"] != -1
    if mask.any():
        loss += criterion(outputs["boredom"][mask], labels["boredom"][mask])
        loss += criterion(outputs["engagement"][mask], labels["engagement"][mask])
        loss += criterion(outputs["confusion"][mask], labels["confusion"][mask])
        loss += criterion(outputs["frustration"][mask], labels["frustration"][mask])
        
    return loss

def compute_accuracy(outputs, labels):
    accs = {}
    for task in ["eye_state", "boredom", "engagement", "confusion", "frustration"]:
        if task == "eye_state":
            preds = torch.argmax(outputs[task], dim=1)
            correct = torch.sum(preds == labels[task]).item()
            accs[task] = (correct, labels[task].size(0))
        else:
            mask = labels[task] != -1
            if mask.any():
                preds = torch.argmax(outputs[task][mask], dim=1)
                correct = torch.sum(preds == labels[task][mask]).item()
                accs[task] = (correct, mask.sum().item())
            else:
                accs[task] = (0, 0)
    return accs

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-out", type=Path, default=Path("models/attention_cnn.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Multi-Task Model on device: {device}")
    print(f"Hyperparameters: Batch Size={args.batch_size}, LR={args.learning_rate}, Epochs={args.epochs}")

    train_transform = transforms.Compose([
        transforms.Resize((180, 180)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    csv_file = args.data_dir / "dataset.csv"
    img_dir = args.data_dir / "images"

    train_dataset = MultiTaskDataset(csv_file, img_dir, "train", transform=train_transform)
    val_dataset = MultiTaskDataset(csv_file, img_dir, "val", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0)

    model = MultiHeadEfficientNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for images, labels in train_pbar:
            images = images.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            
            optimizer.zero_grad()
            
            if scaler:
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    loss = compute_loss(outputs, labels, criterion)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = compute_loss(outputs, labels, criterion)
                loss.backward()
                optimizer.step()
                
            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            train_pbar.set_postfix(loss=(running_loss/total_samples))
            
        scheduler.step()
        
        model.eval()
        val_loss = 0.0
        val_samples = 0
        total_accs = {k: [0,0] for k in ["eye_state", "boredom", "engagement", "confusion", "frustration"]}
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=False)
            for images, labels in val_pbar:
                images = images.to(device)
                labels = {k: v.to(device) for k, v in labels.items()}
                
                outputs = model(images)
                loss = compute_loss(outputs, labels, criterion)
                
                val_loss += loss.item() * images.size(0)
                val_samples += images.size(0)
                
                batch_accs = compute_accuracy(outputs, labels)
                for k, (c, t) in batch_accs.items():
                    total_accs[k][0] += c
                    total_accs[k][1] += t
                    
                val_pbar.set_postfix(loss=(val_loss/val_samples))
                
        v_loss = val_loss / val_samples
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {running_loss/total_samples:.4f} | Val Loss: {v_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        acc_strs = []
        for k, (c, t) in total_accs.items():
            if t > 0:
                acc_strs.append(f"{k}: {c/t:.2f}")
        print("  Accuracies: " + ", ".join(acc_strs))
        
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            epochs_without_improvement = 0
            scripted_model = torch.jit.script(model)
            scripted_model.save(str(args.model_out))
            print(f"  --> Saved improved model to {args.model_out}")
        else:
            epochs_without_improvement += 1
            print(f"  --> No improvement for {epochs_without_improvement} epochs.")
            
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping triggered after {epoch+1} epochs!")
            break

if __name__ == "__main__":
    main()
