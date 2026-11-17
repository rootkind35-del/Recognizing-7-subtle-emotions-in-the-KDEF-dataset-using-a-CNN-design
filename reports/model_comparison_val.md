# So sánh 3 mô hình trên tập `val` khóa của bộ dữ liệu đa văn hóa 3 nguồn

Sinh tự động bởi `python scripts/so_sanh_mo_hinh.py --split val` — cùng 3,990 ảnh, cùng tiền xử lý `ssd_pad_1ch` (SSD square-pad 1,25 + CLAHE), cùng hàm tính độ đo `compute_multicultural_metrics`. Đoán ngẫu nhiên = 14,29%.

Kiểm tra toàn vẹn: y_true của 3 mô hình trùng khớp = **True**, thứ tự ảnh theo manifest = **True**.

## 1. Kết quả chung

| Mô hình | Accuracy | Macro-F1 | Weighted-F1 | ROC-AUC (OvR macro) | Macro-Precision | Macro-Recall | Recall spread | Ảnh đúng |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| KDEF_Stem2Ch_MS4_Symmetric (CNN chính) | 69.37% | 69.47% | 69.21% | 0.9300 | 69.58% | 69.75% | 39.44% | 2,768/3,990 |
| HOG + SVM (LinearSVC) | 42.03% | 38.66% | 41.10% | 0.7739 | 38.07% | 43.65% | 49.56% | 1,677/3,990 |

## 2. Theo từng nguồn dữ liệu

| Mô hình | Nguồn | Số ảnh | Accuracy | Macro-F1 | Weighted-F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| KDEF_Stem2Ch_MS4_Symmetric (CNN chính) | KDEF | 559 | 88.37% | 88.36% | 88.36% | 0.9913 |
| KDEF_Stem2Ch_MS4_Symmetric (CNN chính) | RADIATE | 148 | 86.49% | 83.83% | 86.00% | 0.9751 |
| KDEF_Stem2Ch_MS4_Symmetric (CNN chính) | FER-2013 | 3,283 | 65.37% | 63.12% | 65.17% | 0.9081 |
| HOG + SVM (LinearSVC) | KDEF | 559 | 39.89% | 37.12% | 37.12% | 0.7863 |
| HOG + SVM (LinearSVC) | RADIATE | 148 | 41.89% | 37.86% | 39.78% | 0.7892 |
| HOG + SVM (LinearSVC) | FER-2013 | 3,283 | 42.40% | 35.47% | 41.66% | 0.7533 |

## 3. Theo nhóm người (nguồn gốc)

| Nhóm | Số ảnh | KDEF_Stem2Ch_MS4_Symmetric (CNN chính) Acc / F1 | HOG + SVM (LinearSVC) Acc / F1 |
|---|---:|---:|---:|
| KDEF – Bắc Âu (Thụy Điển) | 559 | 88.4% / 0.884 | 39.9% / 0.371 |
| RADIATE – gốc Phi | 54 | 83.3% / 0.788 | 35.2% / 0.313 |
| RADIATE – gốc Á | 27 | 81.5% / 0.798 | 37.0% / 0.303 |
| RADIATE – gốc Mỹ Latinh | 26 | 76.9% / 0.761 | 38.5% / 0.313 |
| RADIATE – gốc Âu | 41 | 100.0% / 1.000 | 56.1% / 0.537 |
| FER-2013 – Internet | 3,283 | 65.4% / 0.631 | 42.4% / 0.355 |

## 4. Theo giới tính (KDEF + RADIATE, có định danh)

| Giới tính | Số ảnh | KDEF_Stem2Ch_MS4_Symmetric (CNN chính) Acc / F1 | HOG + SVM (LinearSVC) Acc / F1 |
|---|---:|---:|---:|
| Nữ | 304 | 85.9% / 0.856 | 40.8% / 0.386 |
| Nam | 403 | 89.6% / 0.894 | 40.0% / 0.372 |

## 5. Theo góc chụp (phần KDEF, 5 góc)

| Góc | KDEF_Stem2Ch_MS4_Symmetric (CNN chính) | HOG + SVM (LinearSVC) |
|---|---:|---:|
| FL (nghiêng hẳn trái) | 91.0% | 26.1% |
| HL (nửa trái) | 90.2% | 37.5% |
| S (chính diện) | 89.3% | 69.6% |
| HR (nửa phải) | 90.2% | 34.8% |
| FR (nghiêng hẳn phải) | 81.2% | 31.2% |
| \|FL − FR\| | 9.7% | 5.1% |

## 6. Theo từng cảm xúc

### 6.1. F1 từng cảm xúc theo mô hình

| Cảm xúc | Số ảnh | KDEF_Stem2Ch_MS4_Symmetric (CNN chính) | HOG + SVM (LinearSVC) |
|---|---:|---:|---:|
| angry | 538 | 0.6578 | 0.3188 |
| disgust | 141 | 0.8143 | 0.3375 |
| fear | 550 | 0.5112 | 0.2227 |
| happy | 972 | 0.8655 | 0.6273 |
| neutral | 685 | 0.6621 | 0.4215 |
| sad | 724 | 0.5908 | 0.3033 |
| surprise | 380 | 0.7611 | 0.4749 |
| **Macro-F1** | | **0.6947** | **0.3866** |

### 6.2. KDEF_Stem2Ch_MS4_Symmetric (CNN chính): Precision / Recall / F1 / AUC từng cảm xúc

| Cảm xúc | Precision | Recall | F1 | AUC | Recall KDEF | Recall RADIATE | Recall FER-2013 |
|---|---:|---:|---:|---:|---:|---:|---:|
| angry | 0.6712 | 0.6450 | 0.6578 | 0.9167 | 0.8375 | 0.8636 | 0.5986 |
| disgust | 0.8201 | 0.8085 | 0.8143 | 0.9857 | 0.8861 | 0.9545 | 0.5750 |
| fear | 0.5810 | 0.4564 | 0.5112 | 0.8612 | 0.7750 | 0.7273 | 0.3862 |
| happy | 0.8807 | 0.8508 | 0.8655 | 0.9777 | 0.9875 | 1.0000 | 0.8329 |
| neutral | 0.6286 | 0.6993 | 0.6621 | 0.9177 | 0.9250 | 1.0000 | 0.6569 |
| sad | 0.5735 | 0.6091 | 0.5908 | 0.8828 | 0.8625 | 0.7000 | 0.5737 |
| surprise | 0.7153 | 0.8132 | 0.7611 | 0.9683 | 0.9125 | 0.6000 | 0.7931 |

### 6.3. HOG + SVM (LinearSVC): Precision / Recall / F1 / AUC từng cảm xúc

| Cảm xúc | Precision | Recall | F1 | AUC | Recall KDEF | Recall RADIATE | Recall FER-2013 |
|---|---:|---:|---:|---:|---:|---:|---:|
| angry | 0.3411 | 0.2993 | 0.3188 | 0.7267 | 0.3250 | 0.3636 | 0.2913 |
| disgust | 0.2251 | 0.6738 | 0.3375 | 0.8952 | 0.8734 | 0.8182 | 0.2000 |
| fear | 0.2970 | 0.1782 | 0.2227 | 0.6565 | 0.0625 | 0.3182 | 0.1920 |
| happy | 0.6059 | 0.6502 | 0.6273 | 0.8589 | 0.4500 | 0.5333 | 0.6729 |
| neutral | 0.4366 | 0.4073 | 0.4215 | 0.7456 | 0.3750 | 0.4091 | 0.4117 |
| sad | 0.3592 | 0.2624 | 0.3033 | 0.6863 | 0.1375 | 0.0000 | 0.2869 |
| surprise | 0.4000 | 0.5842 | 0.4749 | 0.8482 | 0.5750 | 0.4000 | 0.5931 |

## 7. Quy mô và tốc độ dự đoán

Độ trễ đo theo lô 1 ảnh, tính cả bước chuyển uint8 → tensor → chuẩn hóa → suy luận → softmax (không tính dò mặt), 300 lần sau 20 lần khởi động, fp32.

| Mô hình | Số tham số | Phép nhân-cộng / ảnh | Tệp mô hình | CPU ms/ảnh (FPS) | GPU ms/ảnh (FPS) | Ảnh vào |
|---|---:|---:|---:|---:|---:|---|
| KDEF_Stem2Ch_MS4_Symmetric (CNN chính) | 1,273,005 | 1.325 tỉ | 4.89 MB | 18.59 (54) | 2.06 (487) | xám 160x160, CLAHE=True, mean=0.517243, std=0.264053 |
| HOG + SVM (LinearSVC) | 56,707 | 0.000 tỉ | 0.28 MB | 0.47 (2138) | – (chỉ CPU) | HOG 128x128, ô 8, 9 hướng -> 8100 chiều, C=0.0001, class_weight=balanced |
