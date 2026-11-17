# Nguồn Dữ Liệu, Điều Khoản Sử Dụng & Trích Dẫn (Dataset Attribution)

Tài liệu này ghi nhận nguồn gốc, đường dẫn tải, điều khoản bản quyền, thông báo miễn nhận diện (non-identification notice) và trích dẫn học thuật bắt buộc cho 3 bộ dữ liệu sử dụng trong nhánh `tree2` (nhận diện 7 cảm xúc khuôn mặt đa văn hóa).

---

## 1. KDEF — Karolinska Directed Emotional Faces

- **Trang chủ chính thức**: https://kdef.se/
- **Bản tải cục bộ trong dự án**: `data/official-kdef-dataset-by-ksatrajit0` (nguồn Kaggle: https://www.kaggle.com/datasets/songoku1989/official-kdef-dataset)
- **Quy mô**: 70 người mẫu Thụy Điển (35 nữ, 35 nam, 20–30 tuổi), 4.898 ảnh màu studio (`562×762`, trừ 1 ảnh `633×970`), 5 góc chụp (`FL, HL, S, HR, FR`), 2 phiên chụp (`A, B`). (Sau khi loại 10 ảnh đen hỏng MD5 `320b34032e2133fdfb80a58c953d5c94` ở `train`/`val`, còn 4.888 ảnh hợp lệ.)
- **Giấy phép & Điều khoản sử dụng**:
  - Chỉ sử dụng cho **nghiên cứu khoa học phi thương mại** (strictly non-commercial scientific research only).
  - Không phân phối lại ảnh gốc (no redistribution) khi chưa có văn bản chấp thuận từ Karolinska Institutet, Psychology section; không công khai ảnh gốc trên dịch vụ trực tuyến.
  - Nếu hiển thị ảnh mẫu trong báo cáo/hình minh họa, cần ghi kèm mã định danh ảnh (ví dụ: `F01HAS`).
- **Trích dẫn bắt buộc**:
  - Lundqvist, D., Flykt, A., & Öhman, A. (1998). *The Karolinska Directed Emotional Faces – KDEF* (CD-ROM). Department of Clinical Neuroscience, Psychology section, Karolinska Institutet, Stockholm. ISBN 91-630-7164-9.
  - Goeleven, E., De Raedt, R., Leyman, L., & Verschuere, B. (2008). The Karolinska Directed Emotional Faces: A validation study. *Cognition and Emotion*, 22(6), 1094–1118.
  - Calvo, M. G., & Lundqvist, D. (2008). Facial expressions of emotion (KDEF): Identification under different display-duration conditions. *Behavior Research Methods*, 40(1), 109–115.

---

## 2. RADIATE — The Racially Diverse Affective Expression Face Stimulus Set

- **Trang chủ chính thức**: https://abcdstudy.org/scientists/abcd-fmri-tasks-and-tools/
- **Kho tệp tải về**:
  - `https://abcdstudy.org/newsletter/fMRI_files/RADIATE_JPEGS1.zip`
  - `https://abcdstudy.org/newsletter/fMRI_files/RADIATE_JPEGS2.zip`
  - (`https://abcdstudy.org/newsletter/fMRI_files/RADIATE_BMP.zip` — dự phòng định dạng BMP gốc)
- **Thư mục cục bộ**: `data/radiate/`
- **Quy mô**: 109 người mẫu đa sắc tộc tại Mỹ (18–30 tuổi: 38 gốc Phi `B`, 28 gốc Âu `W`, 22 gốc Á `A`, 20 gốc Mỹ Latinh `H`, 1 nhóm khác `O`; 56 nữ, 53 nam), tổng `1.721` ảnh màu chính diện (`1500×1500` trong `RADIATE_JPEGS1.zip` / `RADIATE_JPEGS2.zip`; tài liệu sườn PDF ghi kích thước chuẩn hóa `450×450`), giữ lại **1.503 ảnh** sau khi loại `218` ảnh `calm` (`CC`, `CO`) để ánh xạ về 7 cảm xúc chuẩn (ghi chú: Bảng 3.1 trong tài liệu sườn PDF liệt kê `1.501`, thực tế các gói nén chính thức chứa `1.721 - 218 = 1.503` ảnh màu hợp lệ; chia theo người mẫu tách biệt sắc tộc: `1.049` `train` / `148` `val` / `306` `test`).
- **Điều khoản sử dụng & Thông báo nguyên văn từ `abcdstudy.org`**:
  > "The tasks and stimuli are available for download here and are free to be used by others for their own research projects. When referencing these tasks be sure to use the provided citations. By downloading these tools, you agree to using them for educational and research purposes only."
  >
  > "By downloading these stimuli, I agree to use them solely for approved institutional research or educational purposes and to not use them in any way to deliberately or inadvertently identify the individuals in the pictures."
- **Trích dẫn bắt buộc**:
  - Conley, M. I., Dellarco, D. V., Rubien-Thomas, E., Cohen, A. O., Cervera, A., Tottenham, N., & Casey, B. J. (2018). The racially diverse affective expression (RADIATE) face stimulus set. *Psychiatry Research*, 270, 1059–1067.
  - Tottenham, N., Tanaka, J. W., Leon, A. C., McCarry, T., Nurse, M., Hare, T. A., Marcus, D. J., Westerlund, A., Casey, B. J., & Nelson, C. (2009). The NimStim set of facial expressions: Judgments from untrained research participants. *Psychiatry Research*, 168(3), 242–249.

---

## 3. FER-2013 — Facial Expression Recognition 2013

- **Trang bộ dữ liệu**: https://www.kaggle.com/datasets/msambare/fer2013
- **API tải về**: `https://www.kaggle.com/api/v1/datasets/download/msambare/fer2013`
- **Thư mục cục bộ**: `data/fer2013/`
- **Quy mô**: 35.887 ảnh xám `48×48` thu thập từ Internet (đa dạng tư thế, ánh sáng, che khuất, độ tuổi, quốc gia; không có mã định danh người mẫu hay nhãn sắc tộc/giới tính): 28.709 `Training`, 3.589 `PublicTest`, 3.589 `PrivateTest`.
- **Giấy phép & Điều khoản sử dụng**:
  - Dữ liệu cuộc thi ICML 2013 Challenges in Representation Learning; phân phối trên Kaggle (`msambare/fer2013`) theo giấy phép **Open Database License (ODbL)** (`Database: Open Database, Contents: Database Contents`), sử dụng cho mục đích nghiên cứu/giáo dục.
- **Trích dẫn bắt buộc**:
  - Goodfellow, I. J., Erhan, D., Carrier, P. L., Courville, A., Mirza, M., Hamner, B., Cukierski, W., Tang, Y., Thaler, D., Lee, D.-H., Zhou, Y., Ramaiah, C., Feng, F., Li, R., Wang, X., Athanasakis, D., Shawe-Taylor, J., Milakov, M., Park, J., Ionescu, R., Popescu, M., Grozea, C., Bergstra, J., Xie, J., Romaszko, L., Xu, B., Chuang, Z., & Bengio, Y. (2013). Challenges in representation learning: A report on three machine learning contests. In *Proceedings of the International Conference on Neural Information Processing (ICONIP)*, pp. 117–124.

---

## 4. Tổng hợp Bộ Dữ Liệu Sạch 3 Nguồn (`data/manifests/multicultural_manifest.csv`)

Theo `data/manifests/manifest_summary.json`, từ **42.288 ảnh thô** ban đầu (`33.116` `train` / `4.297` `val` / `4.875` `test`), sau khi loại `22` ảnh đen/hỏng và `997` bản sao MD5 liên tập khỏi `train`/`val` (tổng loại `1.019` ảnh, giữ nguyên toàn bộ tập `test`), bộ dữ liệu đa văn hóa sạch gồm **41.269 ảnh** (`32.404` `train` / `3.990` `val` / `4.875` `test`, `0` trùng lặp MD5 liên tập, `0/179` trùng người mẫu giữa các tập):

| Nguồn | `train` | `val` | `test` | Tổng sạch |
|---|---:|---:|---:|---:|
| KDEF | 3.349 | 559 | 980 | 4.888 |
| RADIATE | 1.049 | 148 | 306 | 1.503 |
| FER-2013 | 28.006 | 3.283 | 3.589 | 34.878 |
| **Tổng cộng** | **32.404** | **3.990** | **4.875** | **41.269** |
