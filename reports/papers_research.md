# Nghiên cứu tài liệu học thuật — Nhận dạng cảm xúc khuôn mặt đa văn hóa bằng CNN trên KDEF

**Ngày biên soạn:** 2026-10-24
**Bối cảnh đề tài:** KDEF (Karolinska Directed Emotional Faces), 7 lớp (angry, disgust, fear, happy, neutral, sad, surprise), 4.899 ảnh official, chia train 3.359 / val 560 / test 980; CNN tự thiết kế bằng PyTorch; GPU RTX 4060 8GB.

**Phương pháp tra cứu:** Tất cả tài liệu dưới đây được tra cứu trực tuyến trong phiên làm việc này (WebSearch/WebFetch/webReader, tháng 10/2026). Các thông tin chỉ lấy gián tiếp qua bài báo khác được ghi rõ "trích theo ...". Không có số liệu nào bịa ra; những gì không xác minh được trực tuyến được đánh dấu rõ.

---

## A. Danh mục tài liệu

### 1. Lundqvist, Flykt & Öhman (1998) — Bộ dữ liệu KDEF (nguồn gốc)

- **Trích dẫn:** Lundqvist, D., Flykt, A., & Öhman, A. (1998). *The Karolinska Directed Emotional Faces — KDEF*. Phòng tâm lý học, Viện Karolinska, Stockholm. Bản quyền: APA PsycTests, DOI 10.1037/t27732-000.
- **Nguồn tra cứu:** [kdef.se](https://kdef.se) và [doi.apa.org/doi/10.1037/t27732-000) (tra cứu 2026-10-24).
- **Điểm chính:** KDEF gồm **4.900 ảnh** của **70 người** (35 nữ, 35 мужc), mỗi người thể hiện **7 biểu cảm** (happy, sad, angry, fearful, disgusted, surprised, neutral) chụp từ **5 góc** (góc trái 30°, trái 45°, chính diện, phải 45°, phải 30°). Độ tuổi người mẫu 20–30, chủ yếu người Sweden (da sáng) — đây là nguồn gốc tính "đa văn hóa hạn chế" của đề tài.
- **Bài học áp dụng:** (1) Con số 4.899 ảnh của đề tài ≈ toàn bộ KDEF official — tức dữ liệu gồm cả 5 góc chụp, không chỉ chính diện; mô hình phải chịu biến thiên góc đầu ~±45°. (2) Chỉ 70 danh tính ⇒ mỗi người đóng góp ~70 ảnh; nếu split không theo người thì ~70 ảnh của cùng một người xuất hiện cả ở train lẫn test → rủi ro rò rỉ nhận dạng người (identity leakage) rất cao. (3) "Đa văn hóa" trên KDEF thực chất là đa góc chụp + 70 danh tính cùng chủng tộc; cần ghi hạn chế này trong báo cáo đề tài.

### 2. Calvo & Lundqvist (2008) — Chuẩn mức người trên KDEF 7 lớp

- **Trích dẫn:** Calvo, M. G., & Lundqvist, D. (2008). Facial expressions of emotion (KDEF): Identification under different display-duration conditions. *Behavior Research Methods*, 40(1), 109–115.
- **Nguồn tra cứu:** [Springer](https://link.springer.com/article/10.3758/BRM.40.1.109), PubMed 18411533 (tra cứu 2026-10-24; đọc qua tóm tắt/kết quả chính, không đọc toàn văn bảng số liệu).
- **Điểm chính:** Người tham gia nhận diện 280 khuôn mặt KDEF (7 lớp, chính diện) ở các thời lượng hiển thị khác nhau (25 ms → không giới hạn). **Happy nhận diện cao nhất (~98%)**; thứ tự giảm dần thường gặp: happy > surprise > sad ≈ neutral > angry ≈ fear ≈ disgust; các cảm xúc tiêu cực giảm mạnh khi thời lượng ngắn. Biểu cảm dễ nhầm lẫn nhất nằm ở nhóm fear/disgust/anger.
- **Bài học áp dụng:** (1) Mức chuẩn của **người** trên KDEF chính diện ở xem tự do không phải 100% — sai số nhầm lẫn giữa các lớp tiêu cực là hiện tượng thật, mô hình không cần (và không nên kỳ vọng) phân tách hoàn hảo fear/disgust. (2) Khi phân tích nhầm lẫn (confusion matrix), kỳ vọng chính: fear↔surprise, fear↔sad, disgust↔anger, angry↔neutral — khớp dữ liệu tâm lý học. (3) Happy gần như luôn là lớp có F1 cao nhất — dùng làm "sanity check".

### 3. Goeleven et al. (2008) — Thẩm định KDEF bởi người quan sát

- **Trích dẫn:** Goeleven, E., De Raedt, R., Leyman, L., & Verschuere, B. (2008). The Karolinska Directed Emotional Faces: A validation study. *Cognition and Emotion*, 22(6), 1094–1118.
- **Nguồn tra cứu:** bản ghi [PhilPapers](https://philpapers.org/rec/GOETKD) / ResearchGate (tra cứu 2026-10-24; **chưa đọc trực tiếp bảng hit-rate trong phụ lục** — giá trị per-emotion chính xác chưa xác minh trực tuyến).
- **Điểm chính (theo bản ghi):** Thẩm định 490 ảnh chính diện KDEF theo hit rate nhận diện, cường độ và kích thích; các ảnh được chọn làm "chuẩn" thường có hit rate ≥ 0,75. Happy cao nhất; fear/disgust hay bị nhầm.
- **Bài học áp dụng:** Nếu đề tài muốn lọc ảnh "chất lượng cao" (ví dụ chỉ dùng ảnh hit-rate cao) thì đây là nguồn; với phạm vi hiện tại (dùng toàn bộ 4.899 ảnh) chỉ cần biết trần hiệu suất thực tế của nhãn gốc khoảng 75–98% tùy lớp — nhãn gốc có噪声, label smoothing càng hợp lý.

### 4. Cha et al. (2026) — LOSO subject-independent trên KDEF: CNN nhỏ thắng CNN lớn, macro-F1

- **Trích dẫn:** Cha, S. E., et al. (2026). Facial Emotion Recognition via Fusion of Deep and Handcrafted Features (đánh giá trên JAFFE, CK+, KDEF). *PMC13416614*.
- **Nguồn tra cứu:** [pmc.ncbi.nlm.nih.gov/articles/PMC13416614](https://pmc.ncbi.nlm.nih.gov/articles/PMC13416614) — đọc toàn văn số liệu qua WebFetch (tra cứu 2026-10-24). Lưu ý: thí nghiệm KDEF dùng **840 ảnh chính diện, 6 lớp** (không có neutral).
- **Điểm chính:**
  - **Hai protocol so sánh trực tiếp trên KDEF:** 5-fold CV phân tầng ngẫu nhiên (subject-dependent) và **LOSO — leave-one-subject-out (subject-independent)**.
  - **Kết quả LOSO trên KDEF (accuracy / macro-F1):** LeNet-5 (CNN thuần): 0,666 / 0,615; LeNet-5 + đặc trưng handcrafted fusion: **0,802 / 0,773**; ResNet-18 CNN thuần: 0,601 / 0,644; MobileNetV2 CNN thuần: 0,780 / 0,748; fusion tốt nhất ~0,795 / 0,763.
  - **Kết quả 5-fold ngẫu nhiên (subject-dependent):** LeNet-5 fusion 0,843; ResNet-18 0,821; MobileNetV2 0,812; **ResNet-152 chỉ 0,599** — mạng quá sâu overfit dữ liệu nhỏ.
  - Random split → LOSO làm giảm ~4–5 điểm phần trăm (0,843 → 0,802 với cùng mô hình).
  - Dùng grayscale 64×64, Dlib crop mặt, Adam + cross-entropy; **báo cáo macro-F1 vì dữ liệu mất cân bằng lớp**.
- **Bài học áp dụng:** (1) Đây là **mốc đối chiếu quan trọng nhất** cho đề tài: subject-independent 6 lớp chính diện trên KDEF, CNN thuần chỉ đạt **60–78%**; các mô hình tốt (có fusion đặc trưng) đạt ~77–80%. (2) **Không dùng backbone quá lớn** cho 3.359 ảnh train — ResNet-152 còn thua LeNet-5. CNN tự thiết kế ~1–3 tầng convolution sâu vừa phải là lựa chọn đúng. (3) Báo cáo macro-F1 cùng accuracy.

### 5. Akhand et al. (2021) — Ảnh hưởng kích thước ảnh, augmentation, và split ngẫu nhiên trên KDEF

- **Trích dẫn:** Akhand, M. A. H., et al. (2021). Facial Emotion Recognition Using Transfer Learning in the DeepNet Framework. *Electronics (MDPI), 10*(9), 1036.
- **Nguồn tra cứu:** [mdpi.com/2079-9292/10/9/1036](https://www.mdpi.com/2079-9292/10/9/1036) — đọc toàn văn qua webReader (tra cứu 2026-10-24).
- **Điểm chính:**
  - Đánh giá trên KDEF (**full 4.900 ảnh, cả 5 góc, 7 lớp**) với **split ngẫu nhiên 10% test (subject-dependent)** + 10-fold CV.
  - **Standard CNN (2 tầng conv, kernel 3×3) huấn luyện từ đầu theo kích thước ảnh vào:** 48×48 → **61,63%**; 64×64 → **69,39%**; **128×128 → 80,81% (tốt nhất)**; 224×224 → 73,46%; 360×360 → 73,87%.
  - **VGG-16 huấn luyện từ đầu trên KDEF: chỉ 23,35%** (50 vòng) — mạng lớn + dữ liệu nhỏ = thất bại; transfer learning VGG-16 (ImageNet) lên tới **96,51%**, fine-tune 98,78%.
  - Augmentation "ít và nhẹ" hiệu quả nhất cho KDEF: **xoay ±10°, scale ×1,1, lật ngang**; augmentation mạnh làm giảm kết quả.
  - Training: Adam, lr 0,0005.
  - Bảng so sánh các phương pháp trước đó trên KDEF (980 ảnh chính diện, split ngẫu nhiên — **trích theo bảng trong bài, chưa đọc nguyên văn các bài gốc**): Liew & Yairi 2015: 82,40% (90/10); Alshamsi et al. 2017: 90,80% (70/30); Ruiz-Garcia et al. 2017 (SCAE+CNN): 92,52% (70/30).
- **Bài học áp dụng:** (1) **Kích thước ảnh vào tối ưu cho CNN tự thiết kế trên KDEF là 128×128** — quá nhỏ (48) mất thông tin, quá lớn (224+) tăng tham số gây overfit. (2) Với 3.359 ảnh train, KHÔNG huấn luyện VGG/ResNet lớn từ đầu; CNN nhỏ + augmentation nhẹ là hướng đúng (đề tài chủ đích dùng CNN tự thiết kế — phù hợp bằng chứng). (3) Số 96,51% của transfer learning là **split ngẫu nhiên theo ảnh** — không so sánh trực tiếp với kết quả subject-independent của đề tài.

### 6. Shin, Kim & Kwon (2016) — Baseline CNN cho FER, histogram equalization, cross-database

- **Trích dẫn:** Shin, M., Kim, M., Kim, D.-S. (2016). Baseline CNN structure analysis for facial expression recognition. *RO-MAN 2016* (arXiv:1611.04251).
- **Nguồn tra cứu:** [arxiv.org/html/1611.04251v1](https://arxiv.org/html/1611.04251v1) — đọc toàn văn HTML qua WebFetch (tra cứu 2026-10-24).
- **Điểm chính:**
  - Huấn luyện trên FER-2013 + SFEW2.0, kiểm thử chéo trên 5 bộ gồm **KDEF** (2.961 ảnh sau lọc).
  - Kiến trúc đơn giản thắng: **3 tầng conv (5×5 → 4×4 → 5×5) + maxpool, dropout mỗi tầng, FC 3072, input crop 42×42** — đạt kết quả ngang mạng phức tạp với một nửa chi phí tính toán.
  - **Histogram equalization là preprocessing tốt nhất** trong 5 phương pháp thử (raw, HE, isotropic smoothing, DCT norm, DoG).
  - Augmentation: 5 crop ngẫu nhiên + lật ngang (~10× dữ liệu). SGD momentum 0,9, batch 50, lr cố định 0,005, weight decay 1e-5.
  - **Kết quả trên KDEF (cross-database, train FER2013/SFEW): chỉ 50,66%** — thấp nhất trong 5 bộ test do ảnh studio độ phân giải cao khác biệt nhiều với ảnh in-the-wild.
- **Bài học áp dụng:** (1) Kiến trúc CNN "gầy" nhiều dropout đủ sức cho FER 7 lớp — củng cố hướng CNN tự thiết kế. (2) Cân nhắc **histogram equalization** (hoặc CLAHE) làm bước chuẩn hóa trong preprocessing pipeline. (3) Con số 50,66% cho thấy giới hạn thấp của bài toán: nếu kết quả đề tài thấp bất thường (gần 50% với 7 lớp) thì có thể mô hình đang không học được gì thêm ngoài prior lớp; ngược lại, kết quả > 90% với split subject-independent là dấu hiệu **nghi ngờ rò rỉ dữ liệu** chứ không phải thành tích.

### 7. Szegedy et al. (2016) — Label smoothing

- **Trích dẫn:** Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the Inception Architecture for Computer Vision. *CVPR 2016* (arXiv:1512.00567).
- **Nguồn tra cứu:** xác nhận qua kết quả tìm kiếm liên quan (PMC11057568, Ultralytics glossary trích dẫn nguồn gốc — tra cứu 2026-10-24; **chưa mở trực tiếp file arXiv 1512.00567**).
- **Điểm chính:** Giới thiệu label smoothing: thay target one-hot bằng hỗn hợp (1−ε)·one-hot + ε/K (K = số lớp), giúp mô hình ít over-confident, cải thiện generalization và calibration.
- **Bài học áp dụng:** Với 3.359 ảnh train, 7 lớp và nhãn gốc có噪声 (mức người chỉ 75–98% tùy lớp — mục 2, 3), dùng **CrossEntropyLoss(label_smoothing=0,1)** của PyTorch là cách chính quy hóa gần như miễn phí, giảm overfitting FC cuối.

### 8. Loshchilov & Hutter (2019) — AdamW (decoupled weight decay)

- **Trích dẫn:** Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR 2019* (OpenReview Bkg6RiCqY7).
- **Nguồn tra cứu:** [openreview.net/forum?id=Bkg6RiCqY7](https://openreview.net/forum?id=Bkg6RiCqY7) (tra cứu 2026-10-24).
- **Điểm chính:** Tách weight decay khỏi gradient (thay vì cộng L2 vào loss như Adam chuẩn) tạo ra AdamW — generalization tốt hơn Adam+L2, đã thành mặc định trong PyTorch/torch.optim.
- **Bài học áp dụng:** Dùng **AdamW (lr ~1e-3 – 3e-4, weight_decay 1e-4)** thay cho Adam+L2. Với dữ liệu vài nghìn ảnh, weight decay giải耦 là một trong những chỉnh sửa rẻ nhất mang lại lợi ích ổn định.

### 9. Zhong et al. (2020) — Random Erasing (augmentation mô phỏng che khuất)

- **Trích dẫn:** Zhong, Z., Zheng, L., Kang, G., Li, S., & Yang, Y. (2020). Random Erasing Data Augmentation. *AAAI 2020*.
- **Nguồn tra cứu:** [ojs.aaai.org/index.php/AAAI/article/view/7000](https://ojs.aaai.org/index.php/AAAI/article/view/7000/6854) + tài liệu torchvision `RandomErasing` (tra cứu 2026-10-24).
- **Điểm chính:** Xóa ngẫu nhiên một vùng chữ nhật (thay bằng giá trị ngẫu nhiên/trung bình) trong lúc huấn luyện; được kiểm chứng trên person re-ID, nhận diện khuôn mặt và nhận dạng tinh — giảm overfitting, tăng độ bền với che khuất (mắt kính, tóc che, tay che mặt — yếu tố "đa văn hóa" như khẩu trang, hijab một phần).
- **Bài học áp dụng:** Thêm `torchvision.transforms.RandomErasing(p=0.25)` **sau khi tensor hóa, chỉ áp cho tập train**. Không thay thế được flip/xoay nhẹ nhưng bổ sung tốt; chú ý không erasing quá mạnh trên ảnh khuôn mặt vì mắt/miệng là vùng quyết định biểu cảm.

### 10. Rosenblatt et al. (2024) & Apicella et al. (2025) — Rò rỉ dữ liệu theo chủ thể (subject leakage)

- **Trích dẫn:**
  - Rosenblatt, J. D., et al. (2024). Data leakage inflates prediction performance in connectome-based machine learning. *PMC10901797* (diễn đạt "subject leakage" — trùng lặp người giữa train/test).
  - Apicella, A., et al. (2025). Don't push the button! Exploring data leakage risks in machine learning. *Artificial Intelligence Review* (Springer, s10462-025-11326-3).
- **Nguồn tra cứu:** [pmc.ncbi.nlm.nih.gov/articles/PMC10901797](https://pmc.ncbi.nlm.nih.gov/articles/PMC10901797) và [link.springer.com/article/10.1007/s10462-025-11326-3](https://link.springer.com/article/10.1007/s10462-025-11326-3) (tra cứu 2026-10-24; đọc qua tóm tắt kết quả tìm kiếm, **chưa đọc toàn văn**).
- **Điểm chính:** Hiệu suất bị thổi phồng khi (a) cùng chủ thể xuất hiện ở cả train và test ("subject/overlap leakage"); (b) augmentation/oversampling thực hiện **trước khi** chia split; (c) chọn mô hình/hparam dựa trên tập test. Hậu quả được định lượng là tăng ảo hiệu năng dự đoán.
- **Bài học áp dụng (quan trọng nhất về phương pháp luận cho đề tài):**
  1. **Chia train/val/test theo ID người** (mỗi người nằm trọn trong đúng một split). KDEF có 70 người ⇒ ví dụ 48/7/15 người hoặc 47/7/14 (tỉ lệ ảnh sẽ xấp xỉ 68/11/20 như hiện tại nếu phân tầng theo lớp trên từng split). Repo hiện có sẵn `reports/check_leakage.py` — chạy để xác nhận không trùng ID giữa các split.
  2. Augmentation chỉ áp vào **train**, sau khi DataLoader lấy mẫu (không sinh ảnh tăng cường rồi mới chia).
  3. Chỉ "mở" tập test một lần cuối; chọn mô hình và early stopping theo **macro-F1 trên val**.
  4. Khi so sánh với bài báo khác, luôn ghi rõ protocol: subject-dependent (random) hay subject-independent (LOSO/theo người) — chênh nhau ~5–15 điểm phần trăm trên KDEF (so sánh mục 4 vs mục 5).

### 11. Lin et al. (2017) — Focal loss (tham khảo cho mất cân bằng lớp)

- **Trích dẫn:** Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. *ICCV 2017*.
- **Nguồn tra cứu:** xác nhận gián tiếp qua bài tổng quan imbalance trong FER (PMC12109554, "Lin et al.'s focal loss that dynamically adjusts weights..." — tra cứu 2026-10-24; **chưa mở trực tiếp bài gốc**).
- **Điểm chính:** Focal loss (1−p)^γ giảm trọng số mẫu dễ, α-balancing tăng trọng số lớp thiểu số; phổ biến trong FER mất cân bằng.
- **Bài học áp dụng:** KDEF official gần cân bằng (mỗi người × mỗi cảm xúc), nhưng sau khi chia theo người + lọc ảnh lỗi, val/test có thể lệch nhẹ. **Ưu tiên CE + label smoothing trước**; chỉ cân nhắc class-weight/focal khi thấy F1 một lớp nào đó tụt rõ (ví dụ disgust/fear) và đã kiểm tra confusion matrix.

---

## B. Tổng hợp độ chính xác KDEF theo protocol (để đối chiếu thí nghiệm)

| Nguồn | Protocol | Dữ liệu KDEF | Mô hình | Accuracy |
|---|---|---|---|---|
| Cha et al. 2026 | **LOSO subject-independent** | 840 ảnh chính diện, 6 lớp | CNN thuần (LeNet-5/ResNet-18/MobileNetV2) | 60–78% |
| Cha et al. 2026 | **LOSO subject-independent** | 840 ảnh chính diện, 6 lớp | CNN + handcrafted fusion (tốt nhất) | **77–80%** (macro-F1 0,76–0,79) |
| Cha et al. 2026 | 5-fold random (subject-dependent) | 840 ảnh chính diện, 6 lớp | LeNet-5 fusion / ResNet-152 | 84,3% / 59,9% (mạng lớn overfit) |
| Akhand et al. 2021 | 10% random (subject-dependent) | 4.900 ảnh cả 5 góc, 7 lớp | CNN std 2 tầng, input 48/64/**128**/224 | 61,6% / 69,4% / **80,8%** / 73,5% |
| Akhand et al. 2021 | 10% random (subject-dependent) | 4.900 ảnh, 7 lớp | VGG-16 từ đầu / VGG-16 transfer | 23,4% / 96,5% |
| Ruiz-Garcia et al. 2017 (trích theo Akhand 2021) | 70/30 random | 980 ảnh chính diện, 7 lớp | SCAE + CNN | 92,5% |
| Alshamsi et al. 2017 (trích theo Akhand 2021) | 70/30 random | 980 ảnh chính diện, 7 lớp | CNN | 90,8% |
| Shin et al. 2016 | Cross-database (train FER2013) | 2.961 ảnh KDEF | CNN 3 tầng + HE | 50,7% |
| Calvo & Lundqvist 2008 | Người quan sát, xem tự do | 280 ảnh chính diện, 7 lớp | Người | Happy ~98%; nhóm tiêu cực thấp hơn rõ rệt |

**Kết luận đối chiếu (trả lời yêu cầu số 3):**
- Nếu split hiện tại (3.359/560/980) là **ngẫu nhiên theo ảnh (subject-dependent)**: CNN tự thiết kế nên đạt **~78–88% accuracy** trên test (tham chiếu std CNN 80,8% @128×128 của Akhand trên cùng cỡ dữ liệu; các bài báo cũ 90%+ thường dùng trọng số lớn hơn hoặc chỉ ảnh chính diện). Trên **93%** với split kiểu này mà không dùng transfer learning là bất thường → kiểm tra lại rò rỉ.
- Nếu split là **theo người (subject-independent)** — protocol mà đề tài khai báo theo yêu cầu nghiên cứu: dải kỳ vọng thực tế cho CNN tự thiết kế là **~65–80% accuracy, macro-F1 thấp hơn ~2–5 điểm (~62–77%)**. Mốc tham chiếu trực tiếp gần nhất: Cha et al. 2026 LOSO 6 lớp chính diện thu được 60–80% (mặc dù đề tài có nhiều dữ liệu train hơn — 3.359 ảnh cả 5 góc — nhưng cũng khó hơn vì có lớp neutral và đa góc chụp). Đạt **> 80% subject-independent là kết quả rất tốt; > 90% gần như chắc chắn là dấu hiệu rò rỉ** (vì cả mức người trên ảnh chính diện mới chỉ ~98% cho happy và thấp hơn nhiều cho fear/disgust).
- Macro-F1 kỳ vọng thấp hơn accuracy chủ yếu do fear/disgust (nhầm với sad/anger theo dữ liệu tâm lý học — mục 2).

---

## C. Công thức huấn luyện khuyến nghị (tổng hợp từ các nguồn trên)

**Dữ liệu & preprocessing**
1. **Ảnh vào: 128×128, grayscale (1 kênh)** — điểm tối ưu theo Akhand 2021 (80,8% @128 so với 61,6% @48 và 73,5% @224); grayscale giảm tham số, giảm rò rỉ màu da/ánh sáng đặc trưng người (Shin 2016 dùng grayscale + histogram equalization hiệu quả). Có thể thử CLAHE/hist-eq như biến thể (Shin 2016: preprocessing tốt nhất trong 5 phương pháp).
2. Chuẩn hóa: mean/std theo thống kê tập train (đã có `eda_step2_pixels.py` trong repo để lấy số).
3. **Chia split theo người (subject ID)**, phân tầng theo 7 lớp; augmentation chỉ áp trên train (mục 10).

**Kiến trúc CNN tự thiết kế (PyTorch, ~0,5–2M tham số — vừa 8GB VRAM)**
```
Input 1×128×128
Block1: [Conv3×3, 32] ×2 (mỗi conv: Conv→BN→ReLU)   → MaxPool2×2 → 32×64×64
Block2: [Conv3×3, 64] ×2                              → MaxPool2×2 → 64×32×32
Block3: [Conv3×3, 128] ×2                             → MaxPool2×2 → 128×16×16
Block4: [Conv3×3, 256] ×2                             → MaxPool2×2 → 256×8×8
AdaptiveAvgPool2d(4) hoặc GlobalAvgPool → Flatten
Dropout(0,4) → FC(256) → BN → ReLU → Dropout(0,3) → FC(7)
```
Căn cứ: CNN 2–4 tầng là vùng tối ưu cho ~3.359 ảnh (Akhand 2021: std CNN đạt 80,8%; Cha 2026: LeNet-5-scale thắng ResNet-152; Shin 2016: 3 tầng conv + dropout đủ sức). Tránh mạng sâu >20 tầng khi huấn luyện từ đầu (ResNet-152: 59,9%; VGG-16 từ đầu: 23,4%).

**Huấn luyện**
- Optimizer: **AdamW**, lr = **1e-3** ( giảm còn 3e-4 nếu dao động), weight_decay = **1e-4** (Loshchilov & Hutter 2019; Akhand 2021 dùng Adam 5e-4 thành công trên KDEF).
- Scheduler: **ReduceLROnPlateau (factor 0,5, patience 5) hoặc CosineAnnealingLR** — repo hiện đã dùng ReduceLROnPlateau (git log: b4c270d), giữ nguyên.
- Batch size: **64** (3.359 ảnh → ~52 bước/epoch; Shin 2016 dùng 50).
- Loss: **CrossEntropyLoss(label_smoothing=0,1)** (Szegedy 2016). Chỉ thêm class-weight/focal nếu F1 một lớp tụt rõ (Lin 2017, mục 11).
- Augmentation (chỉ train): **lật ngang p=0,5; xoay ±10°; scale ×1,1 / translate nhẹ; RandomErasing p=0,25** (Akhand 2021: "vừa đủ" augmentation; Zhong 2020). Không dùng biến đổi màu mạnh.
- Số epoch: tối đa **60–80**; **early stopping patience 10 epoch** theo **macro-F1 trên val** (bảng số liệu Debnath 2022 hội tụ ~30 epoch; chuẩn chung chọn mô hình theo macro-F1 — Cha 2026).
- Lưu checkpoint theo macro-F1 val tốt nhất; đánh giá cuối: **accuracy + macro-F1 + confusion matrix** trên test một lần.

**Chỉ số chọn mô hình:** macro-F1 trên val (560 ảnh, ~80/lớp — đủ ổn định); báo cáo kèm per-class F1, đặc biệt fear/disgust.

**Kỳ vọng kết quả (để đối chiếu):** xem mục B — subject-independent: **65–80% acc / 62–77% macro-F1**; subject-dependent random: 78–88% acc.

---

## D. Ghi chú về mức độ xác minh

| Thông tin | Trạng thái |
|---|---|
| KDEF 4.900 ảnh, 70 người, 7 biểu cảm, 5 góc (kdef.se) | Đã xác minh trực tuyến |
| Cha et al. 2026: số liệu LOSO/5-fold KDEF (bảng 13 + bảng 5-fold) | Đã đọc trực tiếp qua WebFetch toàn văn PMC |
| Akhand et al. 2021: bảng input size, VGG từ đầu, augmentation, Adam lr | Đã đọc trực tiếp qua webReader (MDPI) |
| Shin et al. 2016: kiến trúc, HE, 50,66% KDEF | Đã đọc trực tiếp qua WebFetch (arXiv HTML) |
| Calvo & Lundqvist 2008: thứ tự độ khó các lớp | Đọc qua tóm tắt/kết quả chính (Springer/PubMed), chưa đọc toàn văn bảng |
| Goeleven 2008: hit-rate per-emotion cụ thể | **CHƯA xác minh trực tuyến** (chỉ có bản ghi, giá trị số chưa lấy được) |
| Ruiz-Garcia 92,52%, Alshamsi 90,8%, Liew & Yairi 82,4% trên KDEF | **Trích theo bảng so sánh của Akhand 2021**, chưa đọc nguyên văn bài gốc |
| Szegedy 2016 label smoothing, Lin 2017 focal loss | Xác minh gián tiếp qua nguồn thứ cấp trích dẫn đúng bài gốc; chưa mở trực tiếp arXiv/ICCV |
| Loshchilov & Hutter 2019 AdamW | Đã xác minh qua OpenReview |
| Rosenblatt 2024 / Apicella 2025 subject leakage | Đọc qua tóm tắt kết quả tìm kiếm, chưa đọc toàn văn |
| Debnath et al. 2022 (four-layer ConvNet) | Đã đọc qua WebFetch — **lưu ý bài này KHÔNG đánh giá trên KDEF** (KDEF chỉ xuất hiện trong bảng liên quan của họ, trích Hussain & Balaji 88%); số liệu hội tụ ~30 epoch lấy từ bài này |
