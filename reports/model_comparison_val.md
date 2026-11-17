# So sánh 3 mô hình trên tập `val` khóa của bộ dữ liệu đa văn hóa 3 nguồn

Sinh tự động bởi `python scripts/so_sanh_mo_hinh.py --split val` — cùng 3,990 ảnh, cùng tiền xử lý `ssd_pad_1ch` (SSD square-pad 1,25 + CLAHE), cùng hàm tính độ đo `compute_multicultural_metrics`. Đoán ngẫu nhiên = 14,29%.

Kiểm tra toàn vẹn: y_true của 3 mô hình trùng khớp = **True**, thứ tự ảnh theo manifest = **True**.

## 1. Kết quả chung

| Mô hình | Accuracy | Macro-F1 | Weighted-F1 | ROC-AUC (OvR macro) | Macro-Precision | Macro-Recall | Recall spread | Ảnh đúng |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| KDEF_Stem2Ch_MS4_Symmetric (CNN chính) | 26.07% | 26.35% | 25.48% | 0.6964 | 41.21% | 32.13% | 52.18% | 1,040/3,990 |

## 2. Theo từng nguồn dữ liệu

| Mô hình | Nguồn | Số ảnh | Accuracy | Macro-F1 | Weighted-F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| KDEF_Stem2Ch_MS4_Symmetric (CNN chính) | KDEF | 559 | 91.23% | 91.27% | 91.28% | 0.9937 |
| KDEF_Stem2Ch_MS4_Symmetric (CNN chính) | RADIATE | 148 | 35.14% | 30.02% | 30.61% | 0.8145 |
| KDEF_Stem2Ch_MS4_Symmetric (CNN chính) | FER-2013 | 3,283 | 14.56% | 10.89% | 11.88% | 0.5782 |

## 3. Theo nhóm người (nguồn gốc)

| Nhóm | Số ảnh | KDEF_Stem2Ch_MS4_Symmetric (CNN chính) Acc / F1 |
|---|---:|---:|
| KDEF – Bắc Âu (Thụy Điển) | 559 | 91.2% / 0.913 |
| RADIATE – gốc Phi | 54 | 27.8% / 0.179 |
| RADIATE – gốc Á | 27 | 33.3% / 0.354 |
| RADIATE – gốc Mỹ Latinh | 26 | 42.3% / 0.310 |
| RADIATE – gốc Âu | 41 | 41.5% / 0.392 |
| FER-2013 – Internet | 3,283 | 14.6% / 0.109 |

## 4. Theo giới tính (KDEF + RADIATE, có định danh)

| Giới tính | Số ảnh | KDEF_Stem2Ch_MS4_Symmetric (CNN chính) Acc / F1 |
|---|---:|---:|
| Nữ | 304 | 73.0% / 0.735 |
| Nam | 403 | 84.4% / 0.847 |

## 5. Theo góc chụp (phần KDEF, 5 góc)

| Góc | KDEF_Stem2Ch_MS4_Symmetric (CNN chính) |
|---|---:|
| FL (nghiêng hẳn trái) | 91.0% |
| HL (nửa trái) | 89.3% |
| S (chính diện) | 89.3% |
| HR (nửa phải) | 95.5% |
| FR (nghiêng hẳn phải) | 91.1% |
| \|FL − FR\| | 0.1% |

## 6. Theo từng cảm xúc

### 6.1. F1 từng cảm xúc theo mô hình

| Cảm xúc | Số ảnh | KDEF_Stem2Ch_MS4_Symmetric (CNN chính) |
|---|---:|---:|
| angry | 538 | 0.2630 |
| disgust | 141 | 0.2671 |
| fear | 550 | 0.2882 |
| happy | 972 | 0.2205 |
| neutral | 685 | 0.2181 |
| sad | 724 | 0.2678 |
| surprise | 380 | 0.3198 |
| **Macro-F1** | | **0.2635** |

### 6.2. KDEF_Stem2Ch_MS4_Symmetric (CNN chính): Precision / Recall / F1 / AUC từng cảm xúc

| Cảm xúc | Precision | Recall | F1 | AUC | Recall KDEF | Recall RADIATE | Recall FER-2013 |
|---|---:|---:|---:|---:|---:|---:|---:|
| angry | 0.1710 | 0.5688 | 0.2630 | 0.6213 | 0.8625 | 0.3636 | 0.5252 |
| disgust | 0.1679 | 0.6525 | 0.2671 | 0.8641 | 0.8354 | 0.6818 | 0.2750 |
| fear | 0.2542 | 0.3327 | 0.2882 | 0.6712 | 0.9250 | 0.7273 | 0.2076 |
| happy | 0.7056 | 0.1307 | 0.2205 | 0.7055 | 0.9875 | 0.2333 | 0.0476 |
| neutral | 0.5536 | 0.1358 | 0.2181 | 0.6260 | 0.9375 | 0.0000 | 0.0309 |
| sad | 0.3397 | 0.2210 | 0.2678 | 0.6463 | 0.9125 | 0.1500 | 0.1346 |
| surprise | 0.6930 | 0.2079 | 0.3198 | 0.7402 | 0.9250 | 0.3000 | 0.0069 |

## 7. Quy mô và tốc độ dự đoán

Độ trễ đo theo lô 1 ảnh, tính cả bước chuyển uint8 → tensor → chuẩn hóa → suy luận → softmax (không tính dò mặt), 300 lần sau 20 lần khởi động, fp32.

| Mô hình | Số tham số | Phép nhân-cộng / ảnh | Tệp mô hình | CPU ms/ảnh (FPS) | GPU ms/ảnh (FPS) | Ảnh vào |
|---|---:|---:|---:|---:|---:|---|
| KDEF_Stem2Ch_MS4_Symmetric (CNN chính) | 1,273,005 | 1.325 tỉ | 4.89 MB | 16.20 (62) | 1.76 (569) | xám 160x160, CLAHE=True, mean=0.407255, std=0.209657 |
