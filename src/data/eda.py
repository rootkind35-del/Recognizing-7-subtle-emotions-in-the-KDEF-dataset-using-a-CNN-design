import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_kdef_distribution(data_dir: str):
    """
    Thống kê số lượng mẫu cho 7 lớp cảm xúc của tập KDEF.
    Các lớp bao gồm: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.
    """
    if not os.path.exists(data_dir):
        print(f"Thư mục {data_dir} không tồn tại.")
        return

    # KDEF splits (train, val, test)
    splits = ["train", "val", "test"]
    stats = []

    for split in splits:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue

        # Liệt kê các thư mục con tương ứng với từng lớp cảm xúc
        emotions = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        
        for emotion in emotions:
            img_paths = glob.glob(os.path.join(split_dir, emotion, "*.jpg"))
            stats.append({
                "split": split,
                "emotion": emotion,
                "count": len(img_paths)
            })

    if not stats:
        print("Không tìm thấy dữ liệu ảnh nào.")
        return

    df = pd.DataFrame(stats)
    print("Thống kê số lượng ảnh:")
    print(df.groupby(["split", "emotion"]).sum())

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="emotion", y="count", hue="split")
    plt.title("Phân phối dữ liệu KDEF theo các lớp cảm xúc")
    plt.xlabel("Cảm xúc")
    plt.ylabel("Số lượng ảnh")
    
    output_path = os.path.join(os.path.dirname(data_dir), "emotion_distribution.png")
    plt.savefig(output_path)
    print(f"Đã lưu biểu đồ tại: {output_path}")

if __name__ == "__main__":
    # Đường dẫn cứng tạm thời đến dataset (có thể đổi thành tham số CLI)
    kdef_path = r"F:\Hoc-May\data\balanced-kdef-dataset-by-dolly-prajapati-182"
    analyze_kdef_distribution(kdef_path)
