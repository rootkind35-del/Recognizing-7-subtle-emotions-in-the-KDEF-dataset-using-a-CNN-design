# Trình Bày Chi Tiết Cấu Tạo 3 Mô Hình So Sánh

Tài liệu này trình bày rõ ràng cơ sở lý thuyết và cấu tạo chi tiết của **3 mô hình** được sử dụng trong bài toán nhận dạng 7 cảm xúc khuôn mặt đa văn hóa (KDEF + RADIATE + FER-2013), bao gồm:
1. **Nền tảng VGG và Mô hình CNN do nhóm tinh chỉnh từ kiến trúc VGG (`KDEF_Stem2Ch_MS4_Symmetric`)**
2. **Mô hình học máy truyền thống `HOG + SVM` (`LinearSVC`)**
3. **Mô hình học chuyển giao `ResNet-18` (`IMAGENET1K_V1`)**

---

## 1. VGG là gì? Và Mô hình CNN do nhóm tinh chỉnh dựa trên nền tảng VGG

### 1.1. VGG là gì?
**VGG** (*Visual Geometry Group* — Nhóm Thị giác Máy tính thuộc Đại học Oxford) là họ kiến trúc mạng nơ-ron tích chập (CNN) kinh điển do Karen Simonyan và Andrew Zisserman công bố năm 2014 (*"Very Deep Convolutional Networks for Large-Scale Image Recognition"*).

**Nguyên lý cốt lõi của VGG:**
- **Xếp chồng các bộ lọc nhỏ $3\times3$:** Thay vì sử dụng bộ lọc kích thước lớn ($7\times7$, $11\times11$) như các kiến trúc trước đó (AlexNet), VGG chỉ dùng đồng nhất các tầng tích chập kích thước $3\times3$ (stride 1, padding 1) xếp liên tiếp thành từng khối (block), ngăn cách bởi các tầng gộp cực đại **Max Pooling $2\times2$** (stride 2).
- **Lợi ích toán học của việc chồng $3\times3$:**
  - Chồng 2 tầng $\text{Conv } 3\times3$ có vùng tiếp nhận (receptive field) tương đương 1 tầng $\text{Conv } 5\times5$, nhưng giảm ~28% số tham số và có 2 lần kích hoạt phi tuyến (ReLU/SiLU) giúp tăng năng lực biểu diễn.
  - Chồng 3 tầng $\text{Conv } 3\times3$ có vùng tiếp nhận tương đương 1 tầng $\text{Conv } 7\times7$ với số tham số giảm ~45%.
- **Quy luật tăng kênh kinh điển:** Sau mỗi lần Max Pooling giảm một nửa chiều rộng và chiều cao không gian, số lượng kênh đặc trưng được nhân đôi ($64 \to 128 \to 256 \to 512$).
- **Hạn chế khi áp dụng trực tiếp VGG-16 / VGG-19 nguyên bản vào bài toán FER:**
  - VGG-16 nguyên bản có tới **138 triệu tham số** (phần lớn tập trung ở 3 tầng kết nối đầy đủ Fully-Connected $7\times7\times512 \to 4096 \to 4096 \to C$ ở cuối mạng), rất nặng (~528 MB), khó huấn luyện từ đầu trên tập dữ liệu khuôn mặt và dễ học thuộc nhân dạng chủ thể (*overfitting*).

---

### 1.2. Cấu tạo mô hình CNN do nhóm tinh chỉnh từ nền tảng VGG (`KDEF_Stem2Ch_MS4_Symmetric`)

Dựa trên nền tảng khối tích chập tuần tự kiểu VGG (xếp chồng các tầng $\text{Conv } 3\times3$, chuẩn hóa theo lô `BatchNorm`, giảm chiều bằng `MaxPool2d(2,2)` và nhân đôi số kênh sau mỗi giai đoạn), nhóm thực hiện **tinh chỉnh lại cấu trúc** cho phù hợp với bài toán nhận dạng cảm xúc khuôn mặt (ảnh xám $1\times160\times160$ đã căn chỉnh khuôn mặt và cân bằng tương phản CLAHE):

1. **Tinh chỉnh tầng đón đầu (Multi-Scale Parallel Stem):** Thay tầng Conv đơn lẻ bằng 2 nhánh song song $\text{Conv } 3\times3$ và $\text{Conv } 5\times5$ để vừa thu nhận chi tiết vi mô (khóe mắt, viền môi) vừa bao quát khối cơ mặt, sau đó nối ghép qua $\text{Conv } 3\times3$ và `MaxPool(2,2)`.
2. **Giữ cấu trúc khối VGG ở Stage 2 & Stage 3, bổ sung chuẩn hóa và kết nối tắt:** Xếp chồng các tầng $\text{Conv } 3\times3 + \text{BatchNorm} + \text{SiLU}$ (2 tầng ở Stage 2; 2–3 tầng ở Stage 3), bổ sung kết nối tắt $\text{Conv } 1\times1$ để truyền dẫn gradient ổn định và tích hợp khối chú ý kênh **ECA-Net** ($\text{Conv1D}_{k=3}$) giúp tập trung vào các kênh chứa đặc trưng biểu cảm.
3. **Tinh chỉnh Stage 4 bằng khối đa tỉ lệ đối xứng (`SymmetricMultiScaleAUBlock`):** Chia thành 4 nhánh song song $\text{Conv } 3\times3$ (đẳng hướng), $\text{Conv } 1\times3$ (ngang — khóe miệng, mí mắt), $\text{Conv } 3\times1$ (dọc — rãnh mũi-má, nhíu mày) và $\text{Conv } 3\times3$ giãn nở $d=2$ (liên kết toàn mặt), rồi trộn lại bằng $\text{Conv } 1\times1$ + ECA-Net. Cấu trúc đối xứng trái/phải giúp mô hình ổn định trên các góc mặt nghiêng trái (`FL`) và phải (`FR`).
4. **Thay thế khối Fully-Connected nặng nề của VGG bằng Facial Strip Pooling:** Sử dụng kết hợp `AdaptiveAvgPool2d((2, 1))` và `AdaptiveMaxPool2d((2, 1))` để tách riêng đặc trưng **nửa trên khuôn mặt (trán – mày – mắt)** và **nửa dưới khuôn mặt (mũi – miệng – cằm)**, giúp giảm mạnh số tham số từ 138 triệu của VGG-16 xuống chỉ còn **1,35 – 3,46 triệu tham số**.
5. **Đầu chiếu phân loại (`FacialProjectionHead`):** Chiếu vector đặc trưng về không gian nhúng chuẩn hóa $L_2$ (phục vụ học tương phản có giám sát `SupConLoss`) trước khi đưa qua tầng tuyến tính phân loại 7 cảm xúc.

#### Bảng cấu tạo chi tiết mô hình CNN nhóm tinh chỉnh (phiên bản `v2_res` / `v3_vgg`):

| Giai đoạn | Thành phần chính | Kích thước đầu ra (`v2_res`: 1,35M / `v3_vgg`: 3,46M) | Vai trò trong nhận dạng cảm xúc |
|---|---|---|---|
| **Đầu vào** | Ảnh xám cắt khuôn mặt SSD + CLAHE | $1 \times 160 \times 160$ | Chuẩn hóa độ sáng và khung khuôn mặt |
| **Stage 1 (Stem)** | Song song $\text{Conv } 3\times3$ + $\text{Conv } 5\times5$ → BN + SiLU → $\text{Conv } 3\times3$ → MaxPool $2\times2$ | $32 \times 80 \times 80$ *(hoặc $48 \times 80 \times 80$)* | Trích xuất cạnh vi mô và khối cơ cơ bản |
| **Stage 2 (VGG Block 2)** | $2 \times [\text{Conv } 3\times3 + \text{BN} + \text{SiLU}]$ + Skip $1\times1$ → MaxPool $2\times2$ | $64 \times 40 \times 40$ *(hoặc $96 \times 40 \times 40$)* | Trích xuất đường nét mắt, mày, môi |
| **Stage 3 (VGG Block 3)** | $2\text{–}3 \times [\text{Conv } 3\times3 + \text{BN} + \text{SiLU}]$ + ECA ($k=3$) + Skip $1\times1$ → MaxPool $2\times2$ | $128 \times 20 \times 20$ *(hoặc $192 \times 20 \times 20$)* | Học tổ hợp vùng cơ mặt, lọc nhiễu danh tính |
| **Stage 4 (AU Block)** | $\text{Conv } 3\times3$ + `SymmetricMultiScaleAUBlock` ($3\times3, 1\times3, 3\times1, d=2$) + Fusion $1\times1$ + ECA + Skip | $256 \times 20 \times 20$ *(hoặc $384 \times 20 \times 20$)* | Nắm bắt Action Units đa hướng và đối xứng góc chụp |
| **Pooling** | `AdaptiveAvgPool2d((2, 1))` $\oplus$ `AdaptiveMaxPool2d((2, 1))` | Vector $1.024$ chiều *(hoặc $1.536$ chiều)* | Tách biệt dải trên (mắt-mày) và dải dưới (mũi-miệng) |
| **Head** | Dropout → Linear ($128$ hoặc $256$) + BN + SiLU → Dropout → Linear ($7$) | $7$ logits cảm xúc | Phân loại 7 lớp cảm xúc + embedding cho SupCon |

---

## 2. Cấu Tạo & Nguyên Lý Mô Hình Đối Chứng 1: `HOG + SVM`

Đây là mô hình đại diện cho hướng tiếp cận **học máy truyền thống (trích xuất đặc trưng thủ công + bộ phân loại tuyến tính)**, gồm 2 khối nối tiếp:

### 2.1. Bộ trích xuất đặc trưng HOG (Histogram of Oriented Gradients — Dalal & Triggs, 2005)
HOG biểu diễn hình dạng khuôn mặt thông qua phân bố hướng của các cạnh (gradient độ sáng) cục bộ:
1. **Tiền xử lý đầu vào:** Ảnh xám khuôn mặt đã căn chỉnh SSD + CLAHE kích thước $160\times160$.
2. **Tính đạo hàm không gian (Gradient):** Với mỗi điểm ảnh $(x, y)$, tính đạo hàm ngang $G_x = I(x+1, y) - I(x-1, y)$ và đạo hàm dọc $G_y = I(x, y+1) - I(x, y-1)$, từ đó suy ra:
   - Độ lớn biên: $M(x, y) = \sqrt{G_x^2 + G_y^2}$
   - Hướng biên (không dấu): $\theta(x, y) = \arctan2(G_y, G_x) \bmod 180^\circ$
3. **Chia ô (Cells) và lập biểu đồ hướng:** Chia ảnh $160\times160$ thành lưới $10 \times 10$ ô (mỗi ô kích thước $16\times16$ pixel). Trong mỗi ô, chia góc $[0^\circ, 180^\circ)$ thành **9 khoảng hướng (bins)** (mỗi bin $20^\circ$) và cộng dồn độ lớn $M$ vào bin tương ứng.
4. **Chuẩn hóa theo khối (Block Normalization — L2-Hys):** Gom nhóm $2\times2$ ô thành một khối trượt (stride 1 ô), tạo thành $9 \times 9 = 81$ khối. Mỗi khối chứa $2 \times 2 \times 9 = 36$ giá trị, được chuẩn hóa bằng chuẩn $L_2$ có cắt ngưỡng (`L2-Hys`) nhằm giảm ảnh hưởng của sự thay đổi độ sáng/độ tương phản.
5. **Vector đặc trưng đầu ra:** Nối toàn bộ 81 khối tạo thành vector đặc trưng cố định kích thước:
   $$81 \times 36 = \mathbf{2.916\text{ chiều}}$$
   Sau đó chuẩn hóa từng chiều về trung bình $0$, độ lệch chuẩn $1$ (`StandardScaler` khớp duy nhất trên tập `train`).

### 2.2. Bộ phân loại tuyến tính SVM (`LinearSVC` — Cortes & Vapnik, 1995)
- **Nguyên lý:** Tìm các siêu phẳng tuyến tính $w_c^\top \phi(x) + b_c = 0$ cho từng lớp $c \in \{1, \dots, 7\}$ (chiến lược One-vs-Rest) sao cho tối đa hóa khoảng cách lề (*maximum margin*) giữa các lớp bằng cách tối thiểu hóa hàm mục tiêu Hinge Loss có điều chuẩn $L_2$:
  $$\min_{w, b} \frac{1}{2}\|w\|_2^2 + C \sum_{i=1}^{N} w_{y_i} \max\big(0, 1 - y_i(w^\top x_i + b)\big)$$
- **Cấu hình tinh chỉnh trên tập `val`:**
  - Tham số phạt $C = 0,0002$ (tránh overfit trên không gian 2.916 chiều).
  - Trọng số lớp `class_weight` điều chỉnh theo nghịch đảo tần suất xuất hiện của 7 cảm xúc để xử lý mất cân bằng dữ liệu.
  - **Hiệu chuẩn xác suất (Platt Scaling):** Do `LinearSVC` chỉ xuất khoảng cách lề (decision score), mô hình lắp thêm bước hiệu chuẩn sigmoid 3-fold (`CalibratedClassifierCV`) trên tập huấn luyện để chuyển sang xác suất hậu nghiệm phục vụ tính độ đo **ROC-AUC**.
- **Quy mô:** Tổng cộng $2.916 \times 7 + 7 = \mathbf{20.419\text{ tham số}}$ (tệp mô hình $0,10\text{ MB}$), tốc độ suy luận CPU cực nhanh (**~0,43 ms/ảnh**, >2.300 FPS).

---

## 3. ResNet là gì? Và Cấu Tạo Mô Hình Đối Chứng 2: `ResNet-18`

### 3.1. ResNet là gì?
**ResNet** (*Residual Network* — Mạng Phần Dư) là kiến trúc mạng học sâu mang tính bước ngoặt do Kaiming He và cộng sự (Microsoft Research) công bố tại hội nghị CVPR 2016 (*"Deep Residual Learning for Image Recognition"*).

**Vấn đề ResNet giải quyết:**
Khi tăng độ sâu của mạng CNN thuần kiểu VGG lên quá nhiều tầng (20, 30, 50 tầng), mạng gặp hiện tượng **triệt tiêu đạo hàm (vanishing gradient)** và **thoái hóa hiệu năng (degradation problem)** — sai số trên cả tập huấn luyện lẫn tập kiểm tra đều tăng lên thay vì giảm xuống.

**Cơ chế Khối Phần Dư (Residual Block):**
Thay vì ép một nhóm tầng tích chập phải học trực tiếp ánh xạ mong muốn $\mathcal{H}(x)$, ResNet thiết kế một **đường kết nối tắt (skip / shortcut connection)** cộng thẳng đầu vào $x$ vào đầu ra của khối tích chập:
$$y = \mathcal{F}(x, \{W_i\}) + x$$
Nhờ đó, các tầng tích chập chỉ cần học **phần dư** $\mathcal{F}(x) = \mathcal{H}(x) - x$. Trong quá trình lan truyền ngược (backpropagation), đạo hàm có thể truyền thẳng qua nhánh $+x$ mà không bị suy giảm theo độ sâu, giúp huấn luyện mạng sâu hội tụ nhanh và ổn định.

### 3.2. Cấu tạo chi tiết của `ResNet-18` (`IMAGENET1K_V1`) trong thực nghiệm
Trong dự án, **ResNet-18** được nạp sẵn trọng số tiền huấn luyện `IMAGENET1K_V1` (đã học trên 1,28 triệu ảnh ImageNet với 1.000 lớp vật thể) và thực hiện **học chuyển giao (Transfer Learning)** sang bài toán 7 cảm xúc khuôn mặt:

- **Bộ chuyển đổi đầu vào:** Ảnh xám khuôn mặt $1\times160\times160$ được lặp thành 3 kênh RGB và chuẩn hóa theo thống kê ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
- **Kiến trúc 18 tầng có trọng số (17 tầng Conv + 1 tầng Linear cuối):**

| Giai đoạn | Cấu tạo khối | Số kênh đầu ra | Kích thước bản đồ đặc trưng (vào $160\times160$) |
|---|---|---|---|
| **Stem (`conv1` + `maxpool`)** | $\text{Conv } 7\times7$ (stride 2) + BN + ReLU → MaxPool $3\times3$ (stride 2) | $64$ | $64 \times 40 \times 40$ |
| **Layer 1 (`2 × BasicBlock`)** | Mỗi khối gồm $2 \times \text{Conv } 3\times3$ + BN + ReLU + Skip connection | $64$ | $64 \times 40 \times 40$ |
| **Layer 2 (`2 × BasicBlock`)** | Khối đầu giảm chiều (stride 2 + Skip $1\times1$), khối sau giữ nguyên | $128$ | $128 \times 20 \times 20$ |
| **Layer 3 (`2 × BasicBlock`)** | Khối đầu giảm chiều (stride 2 + Skip $1\times1$), khối sau giữ nguyên | $256$ | $256 \times 10 \times 10$ |
| **Layer 4 (`2 × BasicBlock`)** | Khối đầu giảm chiều (stride 2 + Skip $1\times1$), khối sau giữ nguyên | $512$ | $512 \times 5 \times 5$ |
| **Pooling & Classifier (`fc`)** | `AdaptiveAvgPool2d((1, 1))` → Dropout → `Linear(512 → 7)` | $7$ | Vector $7$ logits cảm xúc |

- **Quy mô:** Tổng cộng **11.180.103 tham số** (~11,18 triệu tham số, tệp trọng số **42,73 MB** — lớn gấp ~8,2 lần so với CNN `v2_res` và ~3,2 lần so với `v3_vgg`).
- **Quy trình tinh chỉnh (Fine-tuning):** Áp dụng tốc độ học phân tầng (*differential learning rate*: backbone học chậm hơn đầu phân loại), lịch giảm cosine có warmup và trọng số cân bằng lớp trên tập `train`/`val`.
