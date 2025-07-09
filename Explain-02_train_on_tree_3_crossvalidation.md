
# 🌳 Tree Species Detection using Faster R-CNN with Stratified 5-Fold Cross-Validation

This script trains a **Faster R-CNN** object detection model on a forest dataset to classify tree species. It uses **Stratified 5-Fold Cross-Validation** to ensure robust evaluation and handles class imbalance by maintaining label distribution across folds.

---

## 🚀 Key Features

- ⚙️ **Model**: `Faster R-CNN with ResNet-50 FPN`
- 🔁 **Cross-validation**: 5-fold stratified splitting
- 🧠 **Mixed Precision Training (AMP)** for memory and speed optimization
- 🎯 **Early Stopping** and **Learning Rate Scheduling**
- 📊 **Loss curves** per fold are saved for visualization
- 💾 Best and final models are stored per fold

---

## 📁 Directory Structure

```

E:\FASTRCNN\FASTRCNN\dataset2\balanced\_dataset
│
├── images/                               # Training/validation images
├── trainval\_crossvalidation\_labels.csv   # Annotations (bbox + class)
├── checkpoints\_AZ5\_TRY2/                 # Output models and plots
│   ├── fold0\_best.pth
│   ├── fold0\_final.pth
│   ├── fold0\_loss\_curve.png
│   └── ...

```

---

## 🧪 Cross-Validation Setup

- Number of folds: `5`
- Stratified by **major class per image** (using `.mode()` of bounding boxes)
- Each fold runs full training loop with:
  - `MAX_EPOCHS = 20`
  - `PATIENCE = 3` (early stopping)
  - Batch size: `4`
  - Optimizer: SGD + momentum
  - LR scheduler: ReduceLROnPlateau

---

## 🧠 Training Pipeline

1. **Data Loading**
   - Images + labels loaded from `.csv` via `TreeDataset`
   - Bounding boxes + class labels per image

2. **Training Loop**
   - Mixed-precision training via `torch.cuda.amp`
   - Loss computed as sum of all internal Faster R-CNN losses

3. **Validation Loop**
   - Faster R-CNN kept in `train()` mode (for loss return behavior)
   - No gradient calculation during validation

4. **Checkpointing**
   - Saves best model (lowest validation loss)
   - Also saves final model after training ends
   - Loss curves are plotted and saved for each fold

---

## 📉 Example Output – Fold 1 Loss Curve

![Loss Curve](checkpoints_AZ5_TRY2/fold1_loss_curve.png)

---

## 📦 Output Files

| Filename                      | Description                            |
|------------------------------|----------------------------------------|
| `foldN_best.pth`             | Best model (lowest val loss) for fold N |
| `foldN_final.pth`            | Final model at end of training          |
| `foldN_loss_curve.png`       | Training and validation loss plot       |

---

## 📌 Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.10
- torchvision
- pandas
- Pillow
- scikit-learn
- matplotlib
- tqdm

---

## 🗂 Dataset Format

The input `.csv` file must include the following columns:

```

filename, xmin, ymin, xmax, ymax, class

```

Each image may contain multiple rows (bounding boxes) with the same filename.

---

## 📥 Dataset Access

The balanced and weakly labeled dataset used in this training is publicly available:

🔗 [https://yun.ir/9b88b8](https://yun.ir/9b88b8)





