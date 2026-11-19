# So sánh 3 mô hình trên tập `val` khóa của bộ dữ liệu đa văn hóa 3 nguồn

Sinh tự động bởi `python scripts/so_sanh_mo_hinh.py --split val` — cùng 3,990 ảnh, cùng tiền xử lý `ssd_pad_1ch` (SSD square-pad 1,25 + CLAHE), cùng hàm tính độ đo `compute_multicultural_metrics`. Đoán ngẫu nhiên = 14,29%.

Kiểm tra toàn vẹn: y_true của 3 mô hình trùng khớp = **True**, thứ tự ảnh theo manifest = **True**.

## 1. Kết quả chung

| Mô hình | Accuracy | Macro-F1 | Weighted-F1 | ROC-AUC (OvR macro) | Macro-Precision | Macro-Recall | Recall spread | Ảnh đúng |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| KDEF_Stem2Ch_MS4_Symmetric (CNN chính) | 71.63% | 71.05% | 71.49% | 0.9330 | 71.57% | 70.85% | 36.15% | 2,858/3,990 |
| HOG + SVM (LinearSVC) | 45.21% | 42.10% | 43.72% | 0.7891 | 41.48% | 44.90% | 53.16% | 1,804/3,990 |
| ResNet-18 (IMAGENET1K_V1) | 73.01% | 72.84% | 72.85% | 0.9398 | 73.06% | 72.91% | 32.68% | 2,913/3,990 |

## 2. Theo từng nguồn dữ liệu

| Mô hình | Nguồn | Số ảnh | Accuracy | Macro-F1 | Weighted-F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| KDEF_Stem2Ch_MS4_Symmetric (CNN chính) | KDEF | 559 | 90.70% | 90.65% | 90.66% | 0.9953 |
| KDEF_Stem2Ch_MS4_Symmetric (CNN chính) | RADIATE | 148 | 85.14% | 83.93% | 84.96% | 0.9816 |
| KDEF_Stem2Ch_MS4_Symmetric (CNN chính) | FER-2013 | 3,283 | 67.77% | 63.29% | 67.52% | 0.9108 |
| HOG + SVM (LinearSVC) | KDEF | 559 | 41.86% | 38.00% | 37.98% | 0.7850 |
| HOG + SVM (LinearSVC) | RADIATE | 148 | 42.57% | 41.69% | 42.29% | 0.7969 |
| HOG + SVM (LinearSVC) | FER-2013 | 3,283 | 45.90% | 38.82% | 44.46% | 0.7743 |
| ResNet-18 (IMAGENET1K_V1) | KDEF | 559 | 91.59% | 91.57% | 91.57% | 0.9951 |
| ResNet-18 (IMAGENET1K_V1) | RADIATE | 148 | 85.14% | 83.90% | 84.82% | 0.9820 |
| ResNet-18 (IMAGENET1K_V1) | FER-2013 | 3,283 | 69.30% | 66.56% | 69.09% | 0.9203 |

## 3. Theo nhóm người (nguồn gốc)

| Nhóm | Số ảnh | KDEF_Stem2Ch_MS4_Symmetric (CNN chính) Acc / F1 | HOG + SVM (LinearSVC) Acc / F1 | ResNet-18 (IMAGENET1K_V1) Acc / F1 |
|---|---:|---:|---:|---:|
| KDEF – Bắc Âu (Thụy Điển) | 559 | 90.7% / 0.907 | 41.9% / 0.380 | 91.6% / 0.916 |
| RADIATE – gốc Phi | 54 | 79.6% / 0.776 | 37.0% / 0.330 | 87.0% / 0.847 |
| RADIATE – gốc Á | 27 | 81.5% / 0.819 | 44.4% / 0.446 | 77.8% / 0.776 |
| RADIATE – gốc Mỹ Latinh | 26 | 84.6% / 0.834 | 50.0% / 0.520 | 76.9% / 0.760 |
| RADIATE – gốc Âu | 41 | 95.1% / 0.952 | 43.9% / 0.429 | 92.7% / 0.932 |
| FER-2013 – Internet | 3,283 | 67.8% / 0.633 | 45.9% / 0.388 | 69.3% / 0.666 |

## 4. Theo giới tính (KDEF + RADIATE, có định danh)

| Giới tính | Số ảnh | KDEF_Stem2Ch_MS4_Symmetric (CNN chính) Acc / F1 | HOG + SVM (LinearSVC) Acc / F1 | ResNet-18 (IMAGENET1K_V1) Acc / F1 |
|---|---:|---:|---:|---:|
| Nữ | 304 | 83.9% / 0.836 | 39.8% / 0.377 | 87.8% / 0.878 |
| Nam | 403 | 93.8% / 0.937 | 43.7% / 0.405 | 92.1% / 0.920 |

## 5. Theo góc chụp (phần KDEF, 5 góc)

| Góc | KDEF_Stem2Ch_MS4_Symmetric (CNN chính) | HOG + SVM (LinearSVC) | ResNet-18 (IMAGENET1K_V1) |
|---|---:|---:|---:|
| FL (nghiêng hẳn trái) | 90.1% | 27.9% | 86.5% |
| HL (nửa trái) | 89.3% | 42.9% | 94.6% |
| S (chính diện) | 93.8% | 67.9% | 97.3% |
| HR (nửa phải) | 92.9% | 41.1% | 93.8% |
| FR (nghiêng hẳn phải) | 87.5% | 29.5% | 85.7% |
| \|FL − FR\| | 2.6% | 1.5% | 0.8% |

## 6. Theo từng cảm xúc

### 6.1. F1 từng cảm xúc theo mô hình

| Cảm xúc | Số ảnh | KDEF_Stem2Ch_MS4_Symmetric (CNN chính) | HOG + SVM (LinearSVC) | ResNet-18 (IMAGENET1K_V1) |
|---|---:|---:|---:|---:|
| angry | 538 | 0.6624 | 0.3576 | 0.6541 |
| disgust | 141 | 0.7704 | 0.4414 | 0.8114 |
| fear | 550 | 0.5598 | 0.2434 | 0.6123 |
| happy | 972 | 0.8788 | 0.6496 | 0.8841 |
| neutral | 685 | 0.6888 | 0.4299 | 0.6879 |
| sad | 724 | 0.6291 | 0.3382 | 0.6470 |
| surprise | 380 | 0.7846 | 0.4869 | 0.8020 |
| **Macro-F1** | | **0.7105** | **0.4210** | **0.7284** |

### 6.2. KDEF_Stem2Ch_MS4_Symmetric (CNN chính): Precision / Recall / F1 / AUC từng cảm xúc

| Cảm xúc | Precision | Recall | F1 | AUC | Recall KDEF | Recall RADIATE | Recall FER-2013 |
|---|---:|---:|---:|---:|---:|---:|---:|
| angry | 0.6575 | 0.6673 | 0.6624 | 0.9170 | 0.8000 | 0.8182 | 0.6353 |
| disgust | 0.8062 | 0.7376 | 0.7704 | 0.9853 | 0.8608 | 0.9091 | 0.4000 |
| fear | 0.6189 | 0.5109 | 0.5598 | 0.8625 | 0.9000 | 0.7727 | 0.4286 |
| happy | 0.8852 | 0.8724 | 0.8788 | 0.9759 | 0.9875 | 0.9667 | 0.8585 |
| neutral | 0.6430 | 0.7416 | 0.6888 | 0.9230 | 0.9750 | 0.9545 | 0.7015 |
| sad | 0.6339 | 0.6243 | 0.6291 | 0.8951 | 0.9250 | 0.6500 | 0.5849 |
| surprise | 0.7650 | 0.8053 | 0.7846 | 0.9718 | 0.9000 | 0.8000 | 0.7793 |

### 6.3. HOG + SVM (LinearSVC): Precision / Recall / F1 / AUC từng cảm xúc

| Cảm xúc | Precision | Recall | F1 | AUC | Recall KDEF | Recall RADIATE | Recall FER-2013 |
|---|---:|---:|---:|---:|---:|---:|---:|
| angry | 0.3747 | 0.3420 | 0.3576 | 0.7484 | 0.3750 | 0.3636 | 0.3349 |
| disgust | 0.3584 | 0.5745 | 0.4414 | 0.9042 | 0.7975 | 0.5455 | 0.1500 |
| fear | 0.3542 | 0.1855 | 0.2434 | 0.6876 | 0.0125 | 0.3636 | 0.2076 |
| happy | 0.5937 | 0.7171 | 0.6496 | 0.8633 | 0.5500 | 0.5333 | 0.7390 |
| neutral | 0.4277 | 0.4321 | 0.4299 | 0.7610 | 0.4000 | 0.4091 | 0.4374 |
| sad | 0.3791 | 0.3052 | 0.3382 | 0.7004 | 0.2000 | 0.2500 | 0.3205 |
| surprise | 0.4160 | 0.5868 | 0.4869 | 0.8588 | 0.6000 | 0.5000 | 0.5862 |

### 6.4. ResNet-18 (IMAGENET1K_V1): Precision / Recall / F1 / AUC từng cảm xúc

| Cảm xúc | Precision | Recall | F1 | AUC | Recall KDEF | Recall RADIATE | Recall FER-2013 |
|---|---:|---:|---:|---:|---:|---:|---:|
| angry | 0.6635 | 0.6450 | 0.6541 | 0.9205 | 0.8250 | 0.6818 | 0.6101 |
| disgust | 0.8143 | 0.8085 | 0.8114 | 0.9866 | 0.9241 | 0.8636 | 0.5500 |
| fear | 0.6754 | 0.5600 | 0.6123 | 0.8900 | 0.9125 | 0.7727 | 0.4866 |
| happy | 0.8814 | 0.8868 | 0.8841 | 0.9825 | 0.9875 | 1.0000 | 0.8735 |
| neutral | 0.6426 | 0.7401 | 0.6879 | 0.9258 | 0.9000 | 1.0000 | 0.7084 |
| sad | 0.6576 | 0.6367 | 0.6470 | 0.8988 | 0.9000 | 0.8000 | 0.5978 |
| surprise | 0.7792 | 0.8263 | 0.8020 | 0.9744 | 0.9625 | 0.7000 | 0.7931 |

## 7. Quy mô và tốc độ dự đoán

Độ trễ đo theo lô 1 ảnh, tính cả bước chuyển uint8 → tensor → chuẩn hóa → suy luận → softmax (không tính dò mặt), 300 lần sau 20 lần khởi động, fp32.

| Mô hình | Số tham số | Phép nhân-cộng / ảnh | Tệp mô hình | CPU ms/ảnh (FPS) | GPU ms/ảnh (FPS) | Ảnh vào |
|---|---:|---:|---:|---:|---:|---|
| KDEF_Stem2Ch_MS4_Symmetric (CNN chính) | 1,349,165 | 1.351 tỉ | 5.19 MB | 17.16 (58) | 2.15 (466) | xám 160x160, CLAHE=True, mean=0.517243, std=0.264053 |
| HOG + SVM (LinearSVC) | 20,419 | 0.000 tỉ | 0.10 MB | 0.48 (2091) | – (chỉ CPU) | HOG 160x160, ô 16, 9 hướng -> 2916 chiều, C=0.0002, class_weight={0: 0.9373, 1: 1.5803, 2: 0.9297, 3: 0.7717, 4: 0.8718, 5: 0.8796, 6: 1.0296} |
| ResNet-18 (IMAGENET1K_V1) | 11,180,103 | 0.925 tỉ | 42.73 MB | 13.58 (74) | 2.87 (348) | xám 160x160 lặp 3 kênh, chuẩn hóa ImageNet |
