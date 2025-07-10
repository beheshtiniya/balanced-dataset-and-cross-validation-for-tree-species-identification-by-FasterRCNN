
---

### 📊 How to Compute Metrics Using Excel (`Calculating_results.xlsx`)

To evaluate the classification performance , follow the steps below:

---

#### ✅ Step-by-Step Instructions

1. **Locate Confusion Matrix**
   After running `test.py`, a **5×5 confusion matrix** is displayed and used to evaluate the model predictions against the ground-truth labels.

2. **Open the Excel File**
   Open the file:

   ```
   pseudo_labeling_results.xlsx
   ```

3. **Navigate to the Sheet**
   Go to the sheet named:

   ```
   METRICS_AFTER_EVERY_THRESHOLD
   ```

4. **Copy the Confusion Matrix**
   Copy the 5×5 confusion matrix from `test.py` output and paste it into the sheet:

   ```
   Calculating metrics1
   ```

   **Important**: Do **not** include the first row (row 1) of the original matrix if it represents predictions with no matching ground-truth class (false positives only). This row must be excluded from metric computation.

5. **Automatic Calculation**
   Once pasted correctly, the evaluation metrics (Accuracy, Precision, Recall, F1-Score) will be automatically computed in:

   * `Calculating metrics2`
   * `Calculating metrics3`

---

#### ⚠️ Important Note on the First Row of the Matrix

If the **first row (topmost row)** of the confusion matrix contains predictions that have **no corresponding ground-truth labels**, this row **must be excluded** from metric calculations. These values represent **false-positive detections** for which there is no GT equivalent, and including them would **bias** the evaluation.

---

### 📌 Summary

| Sheet Name                      | Description                                                 |
| ------------------------------- | ----------------------------------------------------------- |
| `METRICS_AFTER_EVERY_THRESHOLD` | Stores raw confusion matrices after each run                |
| `Calculating metrics1`          | Paste the confusion matrix here (excluding row 1 if needed) |
| `Calculating metrics2`          | Auto-computes standard metrics                              |
| `Calculating metrics3`          | Additional metric visualization or breakdown                |

ن

