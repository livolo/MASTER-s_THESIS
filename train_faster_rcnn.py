import os
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from fasterrcnn_dataset import FasterRCNNDataset, collate_fn

# ---------------- CONFIG ---------------- #
CSV_PATH = "pseudo_boxes_filtered_clean_v2.csv"
MODEL_BEST = "fasterrcnn_mobile_best.pth"
MODEL_LAST = "fasterrcnn_mobile_last.pth"

EPOCHS = 12
BATCH_SIZE = 1
LR = 1e-4
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", DEVICE)

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------- TRANSFORMS ---------------- #
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
])

# ---------------- DATA ---------------- #
dataset = FasterRCNNDataset(None, CSV_PATH, transform)

n = len(dataset)
print("Total images:", n)

indices = np.random.permutation(n)
n_train = int(0.9 * n)
train_idx = indices[:n_train]
val_idx = indices[n_train:]

train_ds = Subset(dataset, train_idx)
val_ds = Subset(dataset, val_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                        collate_fn=collate_fn, num_workers=2)

print("Train:", len(train_ds), " | Val:", len(val_ds))

# ---------------- MODEL ---------------- #
model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")

# Modify classifier for 2 classes (background + cancer)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ---------------- TRAIN LOOP ---------------- #
best_val = float("inf")
print("Training started:", datetime.now())

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    model.train()
    total_loss = 0

    for i, (images, targets) in enumerate(train_loader, start=1):
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 20 == 0:
            print(f"  Step {i}/{len(train_loader)} | Loss: {total_loss/i:.4f}")

    train_loss = total_loss / len(train_loader)
    print("Train Loss:", round(train_loss, 4))

    # ----- Validation -----
    model.train()
    val_loss = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            val_loss += sum(loss_dict.values()).item()

    val_loss /= len(val_loader)
    print("Val Loss:", round(val_loss, 4))

    # Save last
    torch.save(model.state_dict(), MODEL_LAST)

    # Save best
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), MODEL_BEST)
        print(f"  🔥 Saved BEST model | Val Loss = {val_loss:.4f}")

print("\nTraining finished:", datetime.now())
print("Best Val Loss:", best_val)
