# So sánh 3 mô hình trên tập `test` khóa của bộ dữ liệu đa văn hóa 3 nguồn

Sinh tự động bởi `python scripts/so_sanh_mo_hinh.py --split test --unlock-test` — cùng 4,875 ảnh, cùng tiền xử lý `ssd_pad_1ch` (SSD square-pad 1,25 + CLAHE), cùng hàm tính độ đo `compute_multicultural_metrics`. Đoán ngẫu nhiên = 14,29%.

Kiểm tra toàn vẹn: y_true của 3 mô hình trùng khớp = **True**, thứ tự ảnh theo manifest = **True**.

## 1. Kết quả chung

| Mô hình | Accuracy | Macro-F1 | Weighted-F1 | ROC-AUC (OvR macro) | Macro-Precision | Macro-Recall | Recall spread | Ảnh đúng |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| KDEF_Stem2Ch_MS4_Symmetric (CNN chính) | 73.39% | 72.82% | 73.33% | 0.9450 | 73.81% | 72.30% | 34.84% | 3,578/4,875 |
| HOG + SVM (LinearSVC) | 44.57% | 41.59% | 42.84% | 0.7942 | 41.14% | 43.65% | 56.56% | 2,173/4,875 |
| ResNet-18 (IMAGENET1K_V1) | 74.28% | 73.81% | 74.08% | 0.9492 | 73.77% | 74.08% | 36.12% | 3,621/4,875 |

![So sánh tổng thể 3 mô hình](figures/mc_test_overall_metrics_3models.png)
*Hình 1: Biểu đồ so sánh tổng thể Accuracy, Macro-F1, Weighted-F1 và ROC-AUC của 3 mô hình sau tối ưu trên 4.875 ảnh test khóa.*

## 2. Theo từng nguồn dữ liệu

| Mô hình | Nguồn | Số ảnh | Accuracy | Macro-F1 | Weighted-F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| KDEF_Stem2Ch_MS4_Symmetric (CNN chính) | KDEF | 980 | 86.53% | 86.56% | 86.56% | 0.9810 |
| KDEF_Stem2Ch_MS4_Symmetric (CNN chính) | RADIATE | 306 | 84.97% | 84.37% | 85.05% | 0.9830 |
| KDEF_Stem2Ch_MS4_Symmetric (CNN chính) | FER-2013 | 3,589 | 68.82% | 66.30% | 68.60% | 0.9248 |
| HOG + SVM (LinearSVC) | KDEF | 980 | 41.12% | 38.19% | 38.19% | 0.7725 |
| HOG + SVM (LinearSVC) | RADIATE | 306 | 40.85% | 37.08% | 39.21% | 0.7768 |
| HOG + SVM (LinearSVC) | FER-2013 | 3,589 | 45.83% | 38.08% | 43.98% | 0.7885 |
| ResNet-18 (IMAGENET1K_V1) | KDEF | 980 | 87.35% | 87.44% | 87.44% | 0.9814 |
| ResNet-18 (IMAGENET1K_V1) | RADIATE | 306 | 84.31% | 83.19% | 84.25% | 0.9813 |
| ResNet-18 (IMAGENET1K_V1) | FER-2013 | 3,589 | 69.85% | 67.92% | 69.55% | 0.9303 |

![Độ chính xác theo nguồn](figures/mc_test_accuracy_by_source_3models.png)
*Hình 2: So sánh độ chính xác (Accuracy %) của 3 mô hình trên từng nguồn dữ liệu (KDEF, RADIATE, FER-2013).*

## 3. Theo nhóm người (nguồn gốc)

| Nhóm | Số ảnh | KDEF_Stem2Ch_MS4_Symmetric (CNN chính) Acc / F1 | HOG + SVM (LinearSVC) Acc / F1 | ResNet-18 (IMAGENET1K_V1) Acc / F1 |
|---|---:|---:|---:|---:|
| KDEF – Bắc Âu (Thụy Điển) | 980 | 86.5% / 0.866 | 41.1% / 0.382 | 87.3% / 0.874 |
| RADIATE – gốc Phi | 112 | 78.6% / 0.773 | 33.9% / 0.321 | 80.4% / 0.787 |
| RADIATE – gốc Á | 56 | 83.9% / 0.822 | 48.2% / 0.446 | 83.9% / 0.827 |
| RADIATE – gốc Mỹ Latinh | 56 | 96.4% / 0.964 | 44.6% / 0.373 | 83.9% / 0.789 |
| RADIATE – gốc Âu | 82 | 86.6% / 0.867 | 42.7% / 0.370 | 90.2% / 0.900 |
| FER-2013 – Internet | 3,589 | 68.8% / 0.663 | 45.8% / 0.381 | 69.9% / 0.679 |

## 4. Theo giới tính (KDEF + RADIATE, có định danh)

| Giới tính | Số ảnh | KDEF_Stem2Ch_MS4_Symmetric (CNN chính) Acc / F1 | HOG + SVM (LinearSVC) Acc / F1 | ResNet-18 (IMAGENET1K_V1) Acc / F1 |
|---|---:|---:|---:|---:|
| Nữ | 783 | 83.9% / 0.839 | 39.7% / 0.373 | 84.2% / 0.842 |
| Nam | 503 | 89.7% / 0.896 | 43.1% / 0.409 | 90.5% / 0.904 |

## 5. Theo góc chụp (phần KDEF, 5 góc)

| Góc | KDEF_Stem2Ch_MS4_Symmetric (CNN chính) | HOG + SVM (LinearSVC) | ResNet-18 (IMAGENET1K_V1) |
|---|---:|---:|---:|
| FL (nghiêng hẳn trái) | 84.2% | 27.0% | 85.2% |
| HL (nửa trái) | 87.8% | 42.3% | 89.3% |
| S (chính diện) | 86.2% | 61.7% | 87.8% |
| HR (nửa phải) | 88.8% | 40.8% | 88.8% |
| FR (nghiêng hẳn phải) | 85.7% | 33.7% | 85.7% |
| \|FL − FR\| | 1.5% | 6.6% | 0.5% |

## 6. Theo từng cảm xúc

### 6.1. F1 từng cảm xúc theo mô hình

| Cảm xúc | Số ảnh | KDEF_Stem2Ch_MS4_Symmetric (CNN chính) | HOG + SVM (LinearSVC) | ResNet-18 (IMAGENET1K_V1) |
|---|---:|---:|---:|---:|
| angry | 675 | 0.6569 | 0.3478 | 0.6682 |
| disgust | 239 | 0.7775 | 0.4218 | 0.8089 |
| fear | 712 | 0.5998 | 0.2202 | 0.5911 |
| happy | 1,083 | 0.9001 | 0.6466 | 0.9058 |
| neutral | 810 | 0.7260 | 0.4300 | 0.7398 |
| sad | 778 | 0.6236 | 0.2949 | 0.6306 |
| surprise | 578 | 0.8136 | 0.5501 | 0.8225 |
| **Macro-F1** | | **0.7282** | **0.4159** | **0.7381** |

![F1-Score theo 7 cảm xúc](figures/mc_test_f1_by_emotion_3models.png)
*Hình 3: Biểu đồ F1-Score của từng cảm xúc giữa 3 mô hình trên tập kiểm tra.*

![Ma trận nhầm lẫn 3 mô hình](figures/mc_test_confusion_3models.png)
*Hình 4: Ma trận nhầm lẫn chuẩn hóa của 3 mô hình.*

### 6.2. KDEF_Stem2Ch_MS4_Symmetric (CNN chính): Precision / Recall / F1 / AUC từng cảm xúc

| Cảm xúc | Precision | Recall | F1 | AUC | Recall KDEF | Recall RADIATE | Recall FER-2013 |
|---|---:|---:|---:|---:|---:|---:|---:|
| angry | 0.6815 | 0.6341 | 0.6569 | 0.9231 | 0.6929 | 0.7273 | 0.6090 |
| disgust | 0.8398 | 0.7238 | 0.7775 | 0.9862 | 0.8071 | 0.7727 | 0.4727 |
| fear | 0.6650 | 0.5463 | 0.5998 | 0.8948 | 0.8000 | 0.9091 | 0.4489 |
| happy | 0.9056 | 0.8947 | 0.9001 | 0.9845 | 1.0000 | 0.9219 | 0.8760 |
| neutral | 0.6865 | 0.7704 | 0.7260 | 0.9471 | 0.9500 | 0.8864 | 0.7220 |
| sad | 0.5875 | 0.6645 | 0.6236 | 0.9098 | 0.9214 | 0.8409 | 0.5909 |
| surprise | 0.8007 | 0.8270 | 0.8136 | 0.9693 | 0.8857 | 0.8636 | 0.8053 |

### 6.3. HOG + SVM (LinearSVC): Precision / Recall / F1 / AUC từng cảm xúc

| Cảm xúc | Precision | Recall | F1 | AUC | Recall KDEF | Recall RADIATE | Recall FER-2013 |
|---|---:|---:|---:|---:|---:|---:|---:|
| angry | 0.3810 | 0.3200 | 0.3478 | 0.7455 | 0.2929 | 0.2273 | 0.3360 |
| disgust | 0.3553 | 0.5188 | 0.4218 | 0.9004 | 0.6571 | 0.6364 | 0.0727 |
| fear | 0.3175 | 0.1685 | 0.2202 | 0.6727 | 0.0500 | 0.2727 | 0.1913 |
| happy | 0.5778 | 0.7341 | 0.6466 | 0.8698 | 0.5429 | 0.7031 | 0.7668 |
| neutral | 0.4122 | 0.4494 | 0.4300 | 0.7735 | 0.5357 | 0.3409 | 0.4377 |
| sad | 0.3231 | 0.2712 | 0.2949 | 0.7203 | 0.1929 | 0.1818 | 0.2963 |
| surprise | 0.5127 | 0.5934 | 0.5501 | 0.8775 | 0.6071 | 0.3182 | 0.6034 |

### 6.4. ResNet-18 (IMAGENET1K_V1): Precision / Recall / F1 / AUC từng cảm xúc

| Cảm xúc | Precision | Recall | F1 | AUC | Recall KDEF | Recall RADIATE | Recall FER-2013 |
|---|---:|---:|---:|---:|---:|---:|---:|
| angry | 0.6904 | 0.6474 | 0.6682 | 0.9244 | 0.7429 | 0.7955 | 0.6069 |
| disgust | 0.7866 | 0.8326 | 0.8089 | 0.9836 | 0.8786 | 0.8182 | 0.7273 |
| fear | 0.6399 | 0.5492 | 0.5911 | 0.9032 | 0.7857 | 0.8409 | 0.4621 |
| happy | 0.9013 | 0.9104 | 0.9058 | 0.9871 | 0.9929 | 0.9219 | 0.8965 |
| neutral | 0.6985 | 0.7864 | 0.7398 | 0.9519 | 0.9429 | 0.9773 | 0.7380 |
| sad | 0.6239 | 0.6375 | 0.6306 | 0.9172 | 0.8643 | 0.7273 | 0.5774 |
| surprise | 0.8232 | 0.8218 | 0.8225 | 0.9767 | 0.9071 | 0.7273 | 0.7981 |

## 7. Quy mô và tốc độ dự đoán

Độ trễ đo theo lô 1 ảnh, tính cả bước chuyển uint8 → tensor → chuẩn hóa → suy luận → softmax (không tính dò mặt), 300 lần sau 20 lần khởi động, fp32.

| Mô hình | Số tham số | Phép nhân-cộng / ảnh | Tệp mô hình | CPU ms/ảnh (FPS) | GPU ms/ảnh (FPS) | Ảnh vào |
|---|---:|---:|---:|---:|---:|---|
| KDEF_Stem2Ch_MS4_Symmetric (CNN chính) | 1,349,165 | 1.351 tỉ | 5.19 MB | 17.32 (58) | 2.15 (465) | xám 160x160, CLAHE=True, mean=0.517243, std=0.264053 |
| HOG + SVM (LinearSVC) | 20,419 | 0.000 tỉ | 0.10 MB | 0.43 (2302) | – (chỉ CPU) | HOG 160x160, ô 16, 9 hướng -> 2916 chiều, C=0.0002, class_weight={0: 0.9373, 1: 1.5803, 2: 0.9297, 3: 0.7717, 4: 0.8718, 5: 0.8796, 6: 1.0296} |
| ResNet-18 (IMAGENET1K_V1) | 11,180,103 | 0.925 tỉ | 42.73 MB | 13.89 (72) | 1.99 (504) | xám 160x160 lặp 3 kênh, chuẩn hóa ImageNet |

---

## 8. Nhận xét khoa học khách quan sau tối ưu

1. **Hiệu quả của mô hình CNN đề xuất (`v2_res`, 1,35M tham số):**
   - Sau tối ưu (kết hợp skip-connection Stage 2/3, Avg+Max Strip Pooling `1024 -> 128`, label smoothing và tinh chỉnh siêu tham số), CNN tăng từ 73,05% lên **73,39% Accuracy** (**0,9450 ROC-AUC**, độ lệch góc `|FL - FR|` giảm mạnh từ 3,6% xuống chỉ còn **1,5%**).
   - Trên bộ dữ liệu đa sắc tộc **RADIATE**, CNN đạt **84,97%** (vượt ResNet-18 84,31%), đặc biệt đạt **96,4%** trên nhóm gốc Mỹ Latinh, trong khi chỉ có **1,35 triệu tham số** (nhẹ hơn ~8,3 lần so với 11,18 triệu tham số của ResNet-18) và kích thước tệp chỉ **5,19 MB** (nhỏ hơn ~8,2 lần).
2. **Hiệu quả tối ưu của ResNet-18 (Baseline 2):**
   - Sau khi mở khóa toàn mạng hợp lý kết hợp differential learning rate, warmup cosine và class weighting tối ưu trên `val`, ResNet-18 tăng từ 72,78% lên **74,28% Accuracy** (**73,81% Macro-F1**, **0,9492 ROC-AUC**), đạt độ chính xác cao nhất tổng thể nhờ thừa hưởng biểu diễn từ 1,28 triệu ảnh ImageNet trên tập ảnh đời thực nhiễu cao FER-2013 (69,85%).
3. **Phương pháp truyền thống (HOG + SVM):**
   - Sau khi tối ưu kích thước ô HOG (`160x160`, ô `16`, `2916` chiều) và hiệu chuẩn trọng số lớp, HOG+SVM tăng từ 43,12% lên **44,57% Accuracy**, nhưng vẫn kém hơn ~29% so với các mạng học sâu do hạn chế cố hữu của đặc trưng thủ công.
