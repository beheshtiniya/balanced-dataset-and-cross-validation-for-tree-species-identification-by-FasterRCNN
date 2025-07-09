import os
import csv
import math
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from confusion_matrix import compute_confusion_matrix

# تنظیمات CUDA
CUDA_LAUNCH_BLOCKING = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("🚀 Device:", device)

# مسیر داده‌ها
root_dir = r"E:\FASTRCNN\FASTRCNN\dataset2\balanced_dataset"
combine_result_dir = os.path.join(root_dir, "heckpoints_AZ5_TRY1", "COMBINE_RESULT")

# ایجاد پوشه‌های خروجی
os.makedirs(combine_result_dir, exist_ok=True)

# ---------------------------- 1️⃣ خواندن داده‌های تست ----------------------------
class trDataset(torch.utils.data.Dataset):
    def __init__(self, root, phase):
        self.root = root
        self.phase = phase
        self.targets = pd.read_csv(os.path.join(root, f'{phase}_labels.csv'))
        self.imgs = self.targets['filename'].astype(str)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, r'\FASTRCNN\FASTRCNN\dataset2\balanced_dataset', 'images', self.imgs[idx])
        img = Image.open(img_path).convert('RGB')
        img = F.to_tensor(img)

        filename = self.imgs[idx]
        return img, filename

    def __len__(self):
        return len(self.imgs)

# بارگذاری مجموعه داده
test_dataset = trDataset(root_dir, 'test')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ---------------------------- 2️⃣ اجرای مدل و ذخیره نتایج ----------------------------
output_folder = os.path.join(combine_result_dir, "predicted_boxes")
os.makedirs(output_folder, exist_ok=True)
print(f"📂 مسیر ذخیره خروجی مدل: {output_folder}")

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(1024, 5)
checkpoint = torch.load(os.path.join(root_dir, r"E:\FASTRCNN\FASTRCNN\dataset2\balanced_dataset\checkpoints_AZ5_TRY1", "fold3_best.pth"))

# First inspect what keys are available
print("Checkpoint keys:", checkpoint.keys())

# Then try the appropriate loading method
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
elif 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
else:
    # If checkpoint is the model itself
    model.load_state_dict(checkpoint)

model.to(device)

model.eval()
print("🔍 پردازش تصاویر با مدل...")

with torch.no_grad():
    for images, filenames in tqdm(test_loader, desc="📸 پردازش تصاویر"):
        images = [image.to(device) for image in images]
        out = model(images)

        for i, (filename, prediction) in enumerate(zip(filenames, out)):
            # دریافت نام فایل به‌درستی
            if isinstance(filename, list) or isinstance(filename, tuple):
                filename_str = filename[0]
            else:
                filename_str = str(filename)

            filename_str = os.path.basename(str(filename_str))
            filename_str = os.path.splitext(filename_str)[0]

            # مسیر ذخیره فایل CSV
            output_csv_file = os.path.join(output_folder, f"{filename_str}.csv")

            # پردازش خروجی مدل
            boxes = prediction['boxes'].cpu().numpy()
            labels = prediction['labels'].cpu().numpy()
            scores = prediction['scores'].cpu().numpy()

            # ساختن لیست داده‌ها قبل از تبدیل به DataFrame
            data = [
                [filename_str, label, int(xmin), int(ymin), int(xmax), int(ymax), float(score)]
                for box, label, score in zip(boxes, labels, scores)
                for xmin, ymin, xmax, ymax in [box]
            ]

            # تبدیل به DataFrame
            df = pd.DataFrame(data, columns=['filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'score'])

            # اطمینان از اینکه `filename` به عنوان `str` ذخیره شود
            df['filename'] = df['filename'].astype(str)

            # ذخیره CSV بدون تغییر نام فایل
            df.to_csv(output_csv_file, index=False)

print("✅ پردازش تصاویر به پایان رسید.")

# ---------------------------- 3️⃣ فیلتر کردن باکس‌ها بر اساس IoU با GT ----------------------------
filtered_dots_folder = os.path.join(combine_result_dir, "filtered_predicted_dots")
os.makedirs(filtered_dots_folder, exist_ok=True)

# تابع محاسبه IoU بین دو باکس
def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # محاسبه مختصات ناحیه اشتراک
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # مساحت ناحیه اشتراک
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # مساحت هر باکس
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    # محاسبه IoU
    iou = inter_area / (area1 + area2 - inter_area + 1e-6)  # +1e-6 برای جلوگیری از تقسیم بر صفر
    return iou

# تابع پردازش فایل‌ها
def process_files(boxes_file, gt_df, output_file):
    boxes_df = pd.read_csv(boxes_file)
    output_rows = []

    # استخراج تمام باکس‌های GT مربوط به این تصویر (بر اساس نام فایل)
    filename = os.path.splitext(os.path.basename(boxes_file))[0] + '.tif'  # تطابق نام فایل
    gt_boxes = gt_df[gt_df['filename'] == filename]

    for _, gt_row in gt_boxes.iterrows():
        gt_box = [gt_row['xmin'], gt_row['ymin'], gt_row['xmax'], gt_row['ymax']]

        best_box = None
        best_iou = -1
        best_score = -1

        for _, box_row in boxes_df.iterrows():
            pred_box = [box_row['xmin'], box_row['ymin'], box_row['xmax'], box_row['ymax']]
            score = box_row['score']
            iou = calculate_iou(gt_box, pred_box)

            if iou > 0.5 and (iou > best_iou or (iou == best_iou and score > best_score)):
                best_box = box_row
                best_iou = iou
                best_score = score

        if best_box is not None:
            output_rows.append(best_box.to_dict())

    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(output_file, index=False)

# بارگذاری فایل GT (test_labels.csv)
test_labels_path = os.path.join(root_dir, "test_labels.csv")
gt_df = pd.read_csv(test_labels_path)

# پردازش تمام فایل‌های پیش‌بینی‌شده
for pred_filename in tqdm(os.listdir(output_folder), desc="🔍 فیلتر کردن باکس‌ها بر اساس IoU با GT"):
    if pred_filename.endswith(".csv"):
        boxes_file = os.path.join(output_folder, pred_filename)
        output_file = os.path.join(filtered_dots_folder, pred_filename)
        process_files(boxes_file, gt_df, output_file)

print("✅ فیلتر کردن باکس‌ها بر اساس IoU با GT انجام شد.")

# ---------------------------- 4️⃣ تجمیع فایل‌های CSV ----------------------------
merged_file_path = os.path.join(combine_result_dir, "merged_predictions.csv")

merged_data = []
for csv_file in tqdm(os.listdir(filtered_dots_folder), desc="🔄 تجمیع فایل‌ها"):
    if csv_file.endswith(".csv"):
        file_path = os.path.join(filtered_dots_folder, csv_file)
        df = pd.read_csv(file_path)

        # اصلاح نام فایل‌ها: حذف ".0" و اطمینان از پسوند ".tif"
        df['filename'] = df['filename'].astype(str).str.replace('.0', '', regex=False) + ".tif"

        # حذف ستون "score" اگر وجود داشته باشد
        if 'score' in df.columns:
            df = df.drop(columns=['score'])

        merged_data.append(df)

final_df = pd.concat(merged_data, ignore_index=True)
final_df.to_csv(merged_file_path, index=False)

print(f"✅ فایل تجمیع شده ذخیره شد: {merged_file_path}")

# ---------------------------- 5️⃣ محاسبه متریک‌ها و نمایش نتایج ----------------------------
# تنظیمات CUDA
CUDA_LAUNCH_BLOCKING = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("🚀 Device:", device)

# مسیر داده‌ها
root_dir = r"E:\FASTRCNN\FASTRCNN\dataset2\balanced_dataset"
combine_result_dir = os.path.join(root_dir, "\checkpoints_AZ5_TRY1", "COMBINE_RESULT")

# مسیر فایل‌های مورد نیاز
merged_file_path = os.path.join(combine_result_dir, "merged_predictions.csv")
test_labels_path = os.path.join(root_dir, "test_labels.csv")

# فراخوانی تابع محاسبه کانفیوشن ماتریس
conf_matrix, true_labels, pred_labels = compute_confusion_matrix(test_labels_path, merged_file_path)

# محاسبه متریک‌ها
precision = precision_score(true_labels, pred_labels, average='macro')
recall = recall_score(true_labels, pred_labels, average='macro')
f1 = f1_score(true_labels, pred_labels, average='macro')

# نمایش ماتریس کانفیوشن
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1, 2, 3, 4], yticklabels=[0, 1, 2, 3, 4])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (5x5)")
plt.show()

# نمایش متریک‌ها
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# ذخیره متریک‌ها در فایل CSV
metrics_df = pd.DataFrame({
    "Metric": ["Precision", "Recall", "F1 Score"],
    "Value": [precision, recall, f1]
})
metrics_df.to_csv(os.path.join(combine_result_dir, "classification_metrics.csv"), index=False)
print("✅ پردازش کامل شد.")