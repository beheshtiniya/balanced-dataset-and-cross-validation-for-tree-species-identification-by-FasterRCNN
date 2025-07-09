import os
import time
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# 1. تنظیمات اولیه
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device} | Torch version: {torch.__version__}")

# تنظیمات مسیرها
DATA_ROOT = r'E:\FASTRCNN\FASTRCNN\dataset2\balanced_dataset'
CHECKPOINT_DIR = os.path.join(DATA_ROOT, "checkpoints_AZ5_TRY2")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# 2. تعریف Dataset
class TreeDataset(Dataset):
    def __init__(self, root_dir, dataframe, filenames):
        self.root_dir = root_dir
        self.dataframe = dataframe
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.root_dir, 'images', img_name)
        img = Image.open(img_path).convert('RGB')
        img = F.to_tensor(img)

        img_data = self.dataframe[self.dataframe['filename'] == img_name]
        boxes = img_data[['xmin', 'ymin', 'xmax', 'ymax']].values
        labels = img_data['class'].values

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }

        return img, target, img_name


def collate_fn(batch):
    return tuple(zip(*batch))


# 3. توابع آموزش و ارزیابی با AMP
def train_one_epoch(model, optimizer, data_loader, scaler):
    model.train()
    total_loss = 0
    torch.cuda.empty_cache()

    for images, targets, _ in tqdm(data_loader, desc="Training"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += losses.item()

    return total_loss / len(data_loader)


def validate(model, data_loader):
    model.train()  # NOT eval, because eval mode returns predictions instead of losses
    val_loss = 0
    torch.cuda.empty_cache()

    with torch.no_grad():  # No gradient tracking
        for images, targets, _ in tqdm(data_loader, desc="Validating"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)  # will return dict of losses
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()

    return val_loss / len(data_loader)



# 4. تابع رسم نمودار
def plot_loss(fold, train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='x')
    plt.title(f'Fold {fold} Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(CHECKPOINT_DIR, f"fold{fold}_loss_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss curve saved to: {save_path}")


# 5. اجرای اصلی برنامه
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # برای ویندوز، اختیاری ولی مفید

    # پارامترها
    BATCH_SIZE = 4
    NUM_WORKERS = 4
    MAX_EPOCHS = 20
    PATIENCE = 3
    NUM_FOLDS = 5

    # بارگذاری داده‌ها
    all_data = pd.read_csv(os.path.join(DATA_ROOT, 'trainval_crossvalidation_labels.csv'))
    unique_files = all_data['filename'].unique()
    file_labels = [all_data[all_data['filename'] == fn]['class'].mode()[0] for fn in unique_files]

    # Cross Validation
    start_time = time.time()
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(unique_files, file_labels)):
        print(f"\n{'=' * 40}")
        print(f"Fold {fold + 1}/{NUM_FOLDS}")
        print(f"{'=' * 40}")

        train_dataset = TreeDataset(DATA_ROOT, all_data, unique_files[train_idx])
        val_dataset = TreeDataset(DATA_ROOT, all_data, unique_files[val_idx])

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True, collate_fn=collate_fn,
                                  num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                shuffle=False, collate_fn=collate_fn,
                                num_workers=NUM_WORKERS, pin_memory=True)

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(1024, 5)
        model.to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
        scaler = torch.cuda.amp.GradScaler()

        best_val_loss = float('inf')
        epochs_no_improve = 0
        train_losses, val_losses = [], []

        for epoch in range(MAX_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{MAX_EPOCHS}")

            train_loss = train_one_epoch(model, optimizer, train_loader, scaler)
            val_loss = validate(model, val_loader)
            scheduler.step(val_loss)

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                }, os.path.join(CHECKPOINT_DIR, f"fold{fold}_best.pth"))
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly Stopping at Epoch {epoch + 1}")
                break

        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"fold{fold}_final.pth"))
        plot_loss(fold, train_losses, val_losses)

        print(f"\nGPU Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

    total_time = (time.time() - start_time) / 3600
    print(f"\n{'=' * 40}")
    print(f"Training completed in {total_time:.2f} hours")
    print(f"Results saved in: {CHECKPOINT_DIR}")
    print(f"{'=' * 40}")
