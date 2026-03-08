# train_resnet50_full_upgrade.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "final_dataset"   # must contain train/ val/ test/ with cancer/ non_cancer subfolders
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-4
MODEL_PATH = "resnet50_full_upgrade_best.pth"
PLOT_PREFIX = "resnet50_full_upgrade"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)

# -----------------------------
# TRANSFORMS (strong)
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.05),
    transforms.RandomAffine(degrees=10, translate=(0.05,0.05), scale=(0.9,1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# -----------------------------
# DATASETS
# -----------------------------
train_folder = os.path.join(DATA_DIR, "train")
val_folder   = os.path.join(DATA_DIR, "val")
test_folder  = os.path.join(DATA_DIR, "test")

# quick checks
if not os.path.isdir(train_folder) or not os.path.isdir(val_folder) or not os.path.isdir(test_folder):
    raise FileNotFoundError(f"Expected folders: {DATA_DIR}/train, {DATA_DIR}/val, {DATA_DIR}/test")

train_data = datasets.ImageFolder(train_folder, transform=train_transform)
val_data   = datasets.ImageFolder(val_folder, transform=val_transform)
test_data  = datasets.ImageFolder(test_folder, transform=val_transform)

class_names = train_data.classes
num_classes = len(class_names)
print("Classes:", class_names)

# dataset sizes
print("Train size:", len(train_data), "| Val size:", len(val_data), "| Test size:", len(test_data))

# -----------------------------
# HANDLE IMBALANCE: Oversampling
# -----------------------------
targets = train_data.targets  # list of ints
class_counts = np.bincount(targets)
if len(class_counts) < num_classes:
    # pad if a class has zero examples
    class_counts = np.pad(class_counts, (0, num_classes - len(class_counts)), mode='constant')

print("Class counts:", class_counts)

# sample weights inversely proportional to class frequency
class_weights_for_sampler = 1.0 / class_counts
sample_weights = [class_weights_for_sampler[t] for t in targets]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# -----------------------------
# MODEL: ResNet50 fine-tune (unfreeze layer2/3/4)
# -----------------------------
model = models.resnet50(weights="IMAGENET1K_V2")

# Freeze all then unfreeze deeper layers
for name, param in model.named_parameters():
    param.requires_grad = False
    if ("layer2" in name) or ("layer3" in name) or ("layer4" in name):
        param.requires_grad = True

# Replace final classifier
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
model = model.to(DEVICE)

# -----------------------------
# LOSS: weighted cross entropy (float32)
# -----------------------------
# We want the cancer class to have higher weight.
# Determine mapping: ImageFolder sorts classes alphabetically by folder name unless specified.
# If folders are ["cancer","non_cancer"], index 0 -> cancer, index 1 -> non_cancer
# We construct weight = non_cancer_count / cancer_count for cancer class, and 1.0 for non_cancer.
if num_classes != 2:
    # generic: inverse-frequency weights
    w = 1.0 / class_counts
    weights_tensor = torch.tensor(w, dtype=torch.float32).to(DEVICE)
else:
    # bias cancer class (assuming index 0 is 'cancer')
    cancer_count = float(class_counts[0])
    non_cancer_count = float(class_counts[1])
    w_cancer = non_cancer_count / (cancer_count + 1e-9)
    w_non_cancer = 1.0
    weights_tensor = torch.tensor([w_cancer, w_non_cancer], dtype=torch.float32).to(DEVICE)

print("Loss weights:", weights_tensor.cpu().numpy())
criterion = nn.CrossEntropyLoss(weight=weights_tensor)

# -----------------------------
# OPTIMIZER & SCHEDULER
# -----------------------------
# only optimize parameters that require grad
params_to_optimize = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(params_to_optimize, lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

# -----------------------------
# TRAIN LOOP
# -----------------------------
best_val_acc = 0.0
train_loss_history, val_loss_history = [], []
train_acc_history, val_acc_history = [], []

start_time = datetime.now()
print("Training started at", start_time.strftime("%Y-%m-%d %H:%M:%S"))

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_examples = 0

    for images, labels in train_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        # optional: clip grads
        torch.nn.utils.clip_grad_norm_(params_to_optimize, max_norm=5.0)
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total_examples += images.size(0)

    epoch_train_loss = running_loss / total_examples
    epoch_train_acc = running_corrects / total_examples

    # Validation
    model.eval()
    val_running_loss = 0.0
    val_running_corrects = 0
    val_examples = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            val_running_loss += loss.item() * images.size(0)
            val_running_corrects += torch.sum(preds == labels).item()
            val_examples += images.size(0)

    epoch_val_loss = val_running_loss / (val_examples if val_examples>0 else 1)
    epoch_val_acc = val_running_corrects / (val_examples if val_examples>0 else 1)

    train_loss_history.append(epoch_train_loss)
    val_loss_history.append(epoch_val_loss)
    train_acc_history.append(epoch_train_acc)
    val_acc_history.append(epoch_val_acc)

    # scheduler step on validation loss
    scheduler.step(epoch_val_loss)
    print(f"Scheduler LR: {optimizer.param_groups[0]['lr']}")

    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
          f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")

    # Save best model by val accuracy
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": best_val_acc
        }, MODEL_PATH)
        print("🔥 Saved best model:", MODEL_PATH)

end_time = datetime.now()
print("Training finished at", end_time.strftime("%Y-%m-%d %H:%M:%S"))
print("Total training time:", end_time - start_time)
print("Best validation accuracy:", best_val_acc)

# -----------------------------
# Save loss/acc plots
# -----------------------------
plt.figure()
plt.plot(train_loss_history, label="train_loss")
plt.plot(val_loss_history, label="val_loss")
plt.legend()
plt.title("Loss curve")
plt.savefig(f"{PLOT_PREFIX}_loss.png", bbox_inches='tight')

plt.figure()
plt.plot(train_acc_history, label="train_acc")
plt.plot(val_acc_history, label="val_acc")
plt.legend()
plt.title("Accuracy curve")
plt.savefig(f"{PLOT_PREFIX}_acc.png", bbox_inches='tight')

# -----------------------------
# TEST EVALUATION
# -----------------------------
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
