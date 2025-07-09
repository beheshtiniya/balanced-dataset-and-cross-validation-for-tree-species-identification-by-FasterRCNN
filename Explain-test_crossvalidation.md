
# 🧪 Evaluation of Cross-Validated Faster R-CNN Models on Test Set

This script (`test_crossvalidation.py`) evaluates a **Faster R-CNN** model trained using **Stratified 5-Fold Cross-Validation** on a separate **test set**. It processes predicted outputs, filters them based on IoU, merges valid detections, and computes evaluation metrics.

---

## 🚀 What the Script Does

1. Loads the trained Faster R-CNN model from one fold (e.g., `fold3_best.pth`)
2. Runs inference on the test dataset
3. Filters predictions using IoU > 0.5 with GT boxes
4. Merges all valid predictions into one CSV file
5. Computes classification metrics: Precision, Recall, and F1 Score
6. Visualizes the 5×5 confusion matrix
7. Saves all results to the `COMBINE_RESULT` directory

---

## 📂 Input & Output Structure

```bash
E:\FASTRCNN\FASTRCNN\dataset2\balanced_dataset
│
├── test_labels.csv                            # Ground truth annotations for test set
├── images/                                    # Test images
├── checkpoints_AZ5_TRY1/
│   ├── fold0_best.pth
│   ├── fold1_best.pth
│   ├── fold2_best.pth
│   ├── fold3_best.pth      ← used in current script
│   └── fold4_best.pth
│
└── checkpoints_AZ5_TRY1/COMBINE_RESULT/
    ├── predicted_boxes/                       # Raw predictions per image
    ├── filtered_predicted_dots/               # IoU-filtered boxes
    ├── merged_predictions.csv                 # Final detections
    ├── classification_metrics.csv             # Precision, Recall, F1
````

---

## 🧠 Evaluation Workflow

### 1️⃣ Load Trained Model (Single Fold)

```python
model.load_state_dict(torch.load(... "fold3_best.pth"))
```

* The script only evaluates **one fold at a time** (e.g., fold3).
* Predictions are saved in CSV format for each test image.

---

### 2️⃣ Filter Predictions by IoU with GT

Only predictions with **IoU > 0.5** with ground truth boxes are kept.
Best match per GT box is selected based on IoU and confidence score.

---

### 3️⃣ Merge All Valid Predictions

* All valid detections are merged into a single CSV file.
* Filename strings are normalized and scores are removed.
* Output: `merged_predictions.csv`

---

### 4️⃣ Compute Evaluation Metrics

Using ground truth and merged predictions:

* **Precision**
* **Recall**
* **F1-Score**
* 5×5 **Confusion Matrix** visualized using Seaborn

Results saved to:

```
classification_metrics.csv
```

---

## 🔁 Evaluating All Folds?

> **No**, this script does not automatically evaluate all folds.

By default, only one model checkpoint is loaded (e.g., `fold3_best.pth`).
To evaluate all folds, you must **manually change the checkpoint name and re-run** the script, or modify it as follows:

### 🔄 Example: Auto-Evaluate All Folds

```python
for fold in range(5):
    checkpoint_path = os.path.join(..., f"fold{fold}_best.pth")
    model.load_state_dict(torch.load(checkpoint_path))
    # Perform inference, filtering, merging, and metric calculation for each fold
    # Save results to COMBINE_RESULT/fold{fold}/
```

This way, each fold’s evaluation results will be stored separately and can be analyzed independently or averaged later.

---

## 📥 Dataset Access

The weakly labeled and balanced datasets used in this pipeline are available at:

🔗 [https://yun.ir/9b88b8](https://yun.ir/9b88b8)

---

## 📦 Requirements

* Python ≥ 3.8
* PyTorch ≥ 1.10
* torchvision
* pandas
* Pillow
* scikit-learn
* seaborn
* matplotlib
* tqdm

---

## 📌 Summary of Key Outputs

| Output File                  | Description                           |
| ---------------------------- | ------------------------------------- |
| `predicted_boxes/`           | Raw model predictions per test image  |
| `filtered_predicted_dots/`   | IoU-filtered valid predictions        |
| `merged_predictions.csv`     | Combined valid detections for metrics |
| `classification_metrics.csv` | Precision, Recall, and F1 Score       |
| Confusion Matrix (heatmap)   | Visual summary of predictions         |

---


