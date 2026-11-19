# So sánh Mô hình CNN và 6 Mô hình Học Chuyển Giao (Transfer Learning) trên tập `test` khóa

Sinh tự động bởi `python scripts/so_sanh_mo_hinh.py --split test --unlock-test` — thực hiện theo đúng quy trình và hệ thống biểu đồ của bài báo tham chiếu *"Enhancing Esophageal Cancer Detection Using Transfer Learning Techniques"* (Huynh, Nguyen-Cao et al., TDMU), đánh giá trên cùng **4,875 ảnh** của bộ dữ liệu khuôn mặt đa văn hóa 3 nguồn (`KDEF + RADIATE + FER-2013`), cùng hàm tính độ đo `compute_multicultural_metrics`.

Kiểm tra toàn vẹn: `y_true` của toàn bộ 9 mô hình trùng khớp = **True**, thứ tự ảnh theo manifest = **True**.

## 1. Kiến trúc đề xuất & Quy trình Học chuyển giao (Theo Fig. 1 bài báo tham chiếu)

- **6 Mô hình tiền huấn luyện (Pre-trained Models trên ImageNet-1K):** `VGG16`, `VGG19`, `InceptionV3`, `DenseNet201`, `ResNet50`, `EfficientNetB4`.
- **Cấu trúc tinh chỉnh tầng phân loại (*Customize the Layer of Pre-trained Model*, Mục 3.2 bài báo):**
  - Đóng băng các tầng trích xuất đặc trưng đầu (*frozen initial layers*) để giữ lại đặc trưng cạnh, kết cấu, hình thái cơ bản.
  - Tinh chỉnh các tầng sâu phía sau (*fine-tuned later layers*) kết hợp đầu phân loại tùy chỉnh: `Global Average Pooling` → `FC(128)` → `FC(64)` → `Dropout(0.7)` → `FC(7, Softmax)`.
  - Huấn luyện với tốc độ học `lr = 0.0001` (`1e-4`), tăng cường dữ liệu (*Rotation, Distortion, Flipping*) để xử lý mất cân bằng lớp trong **20 epochs**.
- **Mô hình đề xuất của nhóm (`CNN` — `KDEF_Stem2Ch_MS4_Symmetric`):** Kiến trúc CNN đa tỉ lệ đối xứng chuyên biệt cho nhận dạng cảm xúc khuôn mặt (**1.349.165 tham số**, dung lượng **5,19 MB**).

![Kiến trúc tổng thể hệ thống](figures/paper_fig1_proposed_architecture.png)

*Fig. 1: Sơ đồ kiến trúc tổng thể của nghiên cứu (Proposed architecture of the study).* 

## 2. Biểu đồ Hội tụ Accuracy & Loss qua 20 Epochs (Theo Fig. 2 & Fig. 3 bài báo tham chiếu)

![Biểu đồ Accuracy](figures/paper_fig2_accuracy_curves.png)

*Fig. 2: Training (a) and Validation (b) Accuracy of Models (VGG16, VGG19, DenseNet201, InceptionV3, ResNet50, EfficientNetB4, và CNN).* 

![Biểu đồ Loss](figures/paper_fig3_loss_curves.png)

*Fig. 3: Training (a) and Validation (b) Loss of Models qua 20 epochs.* 

## 3. Bảng So Sánh Hiệu Năng (Table 1 — Performance Comparison of Pre-trained Models & CNN)

| Model | Accuracy | Precision (Macro / Weighted) | Recall (Macro / Weighted) | F1-Score (Macro / Weighted) | Specificity | ROC-AUC | Số tham số | Tệp (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EfficientNetB4 | 55.34% | 53.15% / 55.20% | 53.66% / 55.34% | 53.09% / 55.00% | 92.46% | 0.8683 | 17,787,215 | 68.61 MB |
| InceptionV3 | 57.50% | 56.14% / 57.73% | 56.40% / 57.50% | 56.21% / 57.56% | 92.83% | 0.8529 | 22,056,935 | 84.49 MB |
| VGG19 | 64.47% | 64.15% / 64.70% | 63.84% / 64.47% | 63.96% / 64.56% | 94.00% | 0.8917 | 20,099,143 | 76.70 MB |
| DenseNet201 | 64.51% | 63.97% / 64.78% | 63.65% / 64.51% | 63.78% / 64.61% | 94.01% | 0.8907 | 18,347,911 | 71.32 MB |
| VGG16 | 66.07% | 65.52% / 66.15% | 65.42% / 66.07% | 65.44% / 66.08% | 94.28% | 0.9047 | 14,789,447 | 56.44 MB |
| ResNet50 | 67.02% | 65.97% / 66.93% | 66.27% / 67.02% | 66.07% / 66.93% | 94.44% | 0.9098 | 23,779,399 | 91.03 MB |
| **CNN** | **73.39%** | 73.81% / 73.66% | 72.30% / 73.39% | **72.82%** / 73.33% | 95.50% | 0.9450 | 1,349,165 | 5.19 MB |

![Biểu đồ Table 1](figures/paper_table1_metrics_bar.png)

*Biểu đồ Table 1: Đối sánh trực quan Accuracy, Precision, Recall và F1-Score giữa 6 mô hình Transfer Learning và mô hình CNN.* 


## 4. Ma Trận Nhầm Lẫn Của Các Mô Hình (Fig. 4 — Confusion Matrices)

![Ma trận nhầm lẫn các mô hình](figures/paper_fig4_confusion_matrices.png)

*Fig. 4: Confusion matrices of pre-trained models and proposed CNN.* 


## 5. Bảng Tổng Hợp Toàn Bộ Mô Hình (Bao gồm cả Baselines bổ sung)

| Mô hình | Accuracy | Macro-F1 | Weighted-F1 | Macro-Precision | Macro-Recall | Specificity | ROC-AUC (OvR) | Ảnh đúng |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN (KDEF_Stem2Ch_MS4_Symmetric) | 73.39% | 72.82% | 73.33% | 73.81% | 72.30% | 95.50% | 0.9450 | 3,578/4,875 |
| VGG16 (ImageNet Transfer Learning) | 66.07% | 65.44% | 66.08% | 65.52% | 65.42% | 94.28% | 0.9047 | 3,221/4,875 |
| VGG19 (ImageNet Transfer Learning) | 64.47% | 63.96% | 64.56% | 64.15% | 63.84% | 94.00% | 0.8917 | 3,143/4,875 |
| InceptionV3 (ImageNet Transfer Learning) | 57.50% | 56.21% | 57.56% | 56.14% | 56.40% | 92.83% | 0.8529 | 2,803/4,875 |
| DenseNet201 (ImageNet Transfer Learning) | 64.51% | 63.78% | 64.61% | 63.97% | 63.65% | 94.01% | 0.8907 | 3,145/4,875 |
| ResNet50 (ImageNet Transfer Learning) | 67.02% | 66.07% | 66.93% | 65.97% | 66.27% | 94.44% | 0.9098 | 3,267/4,875 |
| EfficientNetB4 (ImageNet Transfer Learning) | 55.34% | 53.09% | 55.00% | 53.15% | 53.66% | 92.46% | 0.8683 | 2,698/4,875 |
| HOG + SVM (LinearSVC) | 44.57% | 41.59% | 42.84% | 41.14% | 43.65% | 90.59% | 0.7942 | 2,173/4,875 |
| ResNet-18 (IMAGENET1K_V1) | 74.28% | 73.81% | 74.08% | 73.77% | 74.08% | 95.66% | 0.9492 | 3,621/4,875 |

## 6. Theo từng nguồn dữ liệu (KDEF, RADIATE, FER-2013)

| Mô hình | KDEF Acc / F1 (980 ảnh) | RADIATE Acc / F1 (306 ảnh) | FER-2013 Acc / F1 (3,589 ảnh) |
|---|---:|---:|---:|
| CNN | 86.53% / 86.56% | 84.97% / 84.37% | 68.82% / 66.30% |
| VGG16 | 75.61% / 75.49% | 76.14% / 75.42% | 62.61% / 60.24% |
| VGG19 | 72.86% / 73.08% | 72.22% / 71.07% | 61.52% / 59.15% |
| InceptionV3 | 56.12% / 56.21% | 55.88% / 54.25% | 58.01% / 55.43% |
| DenseNet201 | 70.82% / 71.06% | 61.76% / 60.75% | 63.03% / 60.43% |
| ResNet50 | 78.37% / 78.33% | 69.61% / 66.98% | 63.69% / 59.83% |
| EfficientNetB4 | 53.27% / 53.14% | 50.98% / 49.19% | 56.28% / 50.98% |
| HOG + SVM | 41.12% / 38.19% | 40.85% / 37.08% | 45.83% / 38.08% |
| ResNet-18 | 87.35% / 87.44% | 84.31% / 83.19% | 69.85% / 67.92% |

## 7. Theo từng lớp cảm xúc (F1-Score 7 cảm xúc)

| Cảm xúc | Số ảnh | EfficientNetB4 | InceptionV3 | VGG19 | DenseNet201 | VGG16 | ResNet50 | CNN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| angry | 675 | 0.4676 | 0.4788 | 0.5460 | 0.5365 | 0.5779 | 0.5880 | 0.6569 |
| disgust | 239 | 0.4813 | 0.5646 | 0.6983 | 0.6722 | 0.7134 | 0.6988 | 0.7775 |
| fear | 712 | 0.3573 | 0.4251 | 0.5142 | 0.5213 | 0.5083 | 0.5228 | 0.5998 |
| happy | 1,083 | 0.7672 | 0.7882 | 0.8510 | 0.8444 | 0.8663 | 0.8683 | 0.9001 |
| neutral | 810 | 0.5506 | 0.5433 | 0.6171 | 0.6349 | 0.6448 | 0.6651 | 0.7260 |
| sad | 778 | 0.4406 | 0.4800 | 0.5072 | 0.5082 | 0.5221 | 0.5338 | 0.6236 |
| surprise | 578 | 0.6513 | 0.6543 | 0.7434 | 0.7469 | 0.7478 | 0.7478 | 0.8136 |
| **Macro-F1** | | **0.5309** | **0.5621** | **0.6396** | **0.6378** | **0.6544** | **0.6607** | **0.7282** |

## 8. Quy mô tham số và tốc độ suy luận

| Mô hình | Số tham số | Phép nhân-cộng / ảnh | Tệp mô hình | CPU ms/ảnh (FPS) | GPU ms/ảnh (FPS) | Cấu hình đầu vào & Head |
|---|---:|---:|---:|---:|---:|---|
| CNN | 1,349,165 | 1.351 tỉ | 5.19 MB | 16.43 (61) | 2.03 (493) | xám 160x160, CLAHE=True, mean=0.517243, std=0.264053 |
| VGG16 | 14,789,447 | 5.011 tỉ | 56.44 MB | 33.39 (30) | 3.57 (280) | xám 128x128 lặp 3 kênh RGB, chuẩn hóa ImageNet, GAP->128->64->Drop(0.7)->7 |
| VGG19 | 20,099,143 | 6.370 tỉ | 76.70 MB | 41.40 (24) | 4.28 (234) | xám 128x128 lặp 3 kênh RGB, chuẩn hóa ImageNet, GAP->128->64->Drop(0.7)->7 |
| InceptionV3 | 22,056,935 | 0.731 tỉ | 84.49 MB | 24.70 (40) | 11.72 (85) | xám 128x128 lặp 3 kênh RGB, chuẩn hóa ImageNet, GAP->128->64->Drop(0.7)->7 |
| DenseNet201 | 18,347,911 | 1.401 tỉ | 71.32 MB | 71.94 (14) | 25.44 (39) | xám 128x128 lặp 3 kênh RGB, chuẩn hóa ImageNet, GAP->128->64->Drop(0.7)->7 |
| ResNet50 | 23,779,399 | 1.335 tỉ | 91.03 MB | 22.65 (44) | 5.36 (187) | xám 128x128 lặp 3 kênh RGB, chuẩn hóa ImageNet, GAP->128->64->Drop(0.7)->7 |
| EfficientNetB4 | 17,787,215 | 0.493 tỉ | 68.61 MB | 42.28 (24) | 13.48 (74) | xám 128x128 lặp 3 kênh RGB, chuẩn hóa ImageNet, GAP->128->64->Drop(0.7)->7 |
| HOG + SVM | 20,419 | 0.000 tỉ | 0.10 MB | 0.43 (2316) | – (chỉ CPU) | HOG 160x160, ô 16, 9 hướng -> 2916 chiều, C=0.0002 |
| ResNet-18 | 11,180,103 | 0.925 tỉ | 42.73 MB | 12.83 (78) | 1.99 (503) | xám 160x160 lặp 3 kênh, chuẩn hóa ImageNet |
