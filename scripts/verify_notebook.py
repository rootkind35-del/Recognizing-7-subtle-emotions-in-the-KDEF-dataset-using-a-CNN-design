# -*- coding: utf-8 -*-
"""Kiểm tra tự động notebook CNN_KDEF_FER sau khi chạy xong.

Chạy:  python scripts/verify_notebook.py [đường_dẫn_notebook] [đường_dẫn_test_metrics.json]
       (mặc định: notebooks/CNN_KDEF_FER.ipynb và reports/test_metrics.json)

Nội dung kiểm tra:
1. Mọi code cell ĐÃ ĐƯỢC THỰC THI (có execution_count) và KHÔNG có output kiểu error.
2. reports/test_metrics.json tồn tại và đủ các khóa:
   accuracy, macro_f1, weighted_f1, per_class_recall (đủ 7 lớp), recall_spread,
   confusion_matrix (7×7), device, epochs, best_config.
3. Kỷ luật KHÓA TẬP KIỂM TRA (QUY TẮC SỐ 1): trong các code cell TRƯỚC cell chứa chuỗi
   'MỞ KHÓA TẬP TEST', không cell nào chứa đường dẫn thư mục tập kiểm tra
   (dạng 'test' được nhắc/nối như một thành phần path) — trừ cell có '# EDA-cho-phep'
   ở dòng đầu.
4. Nếu tất cả đạt: in đúng MỘT dòng 'VERIFY_OK {json}' với json gồm
   accuracy, macro_f1 (số %), min_class_recall, recall_spread (đơn vị %) — exit 0.
   Ngược lại: in lý do cụ thể từng mục — exit 1.
"""
import json
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows: in được tiếng Việt
except Exception:
    pass

import nbformat  # noqa: E402

GOC_REPO = Path(__file__).resolve().parent.parent
NB_MAC_DINH = GOC_REPO / "notebooks" / "CNN_KDEF_FER.ipynb"
METRICS_MAC_DINH = GOC_REPO / "reports" / "test_metrics.json"

CHUOI_MO_KHOA = "MỞ KHÓA TẬP TEST"
THE_EDA = "# EDA-cho-phep"
# 'test' như một thành phần đường dẫn: "test" / 'test' được nhắc, /test, test/
REG_TEST = re.compile(r"""["']test["']|[/\\]test\b|\btest[/\\]""")

CAC_KHOA_BAT_BUOC = ["accuracy", "macro_f1", "weighted_f1", "per_class_recall",
                     "recall_spread", "confusion_matrix", "device", "epochs", "best_config"]


def kiem_tra_thuc_thi(nb, loi):
    """(1) Mọi code cell có execution_count và không có output lỗi."""
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        if cell.get("execution_count") is None:
            loi.append(f"[1] Cell code thứ {i} CHƯA được thực thi (thiếu execution_count).")
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                loi.append(f"[1] Cell code thứ {i} có output lỗi: "
                           f"{out.get('ename', '?')}: {str(out.get('evalue', ''))[:120]}")


def kiem_tra_metrics(duong_metrics, loi):
    """(2) reports/test_metrics.json tồn tại và đủ khóa."""
    if not Path(duong_metrics).exists():
        loi.append(f"[2] Không tìm thấy file số liệu: {duong_metrics} — hãy chạy xong mục 9 của notebook.")
        return None
    with open(duong_metrics, encoding="utf-8") as f:
        m = json.load(f)
    thieu = [k for k in CAC_KHOA_BAT_BUOC if k not in m]
    if thieu:
        loi.append(f"[2] test_metrics.json thiếu khóa: {thieu}")
        return m
    if len(m["per_class_recall"]) != 7:
        loi.append(f"[2] per_class_recall phải có đủ 7 lớp, thực tế {len(m['per_class_recall'])}.")
    cm = m["confusion_matrix"]
    if not (isinstance(cm, list) and len(cm) == 7 and all(isinstance(r, list) and len(r) == 7 for r in cm)):
        loi.append("[2] confusion_matrix phải là ma trận 7×7.")
    for k in ["accuracy", "macro_f1", "weighted_f1", "recall_spread"]:
        if not isinstance(m[k], (int, float)):
            loi.append(f"[2] Khóa {k} phải là số, thực tế {type(m[k]).__name__}.")
    return m


def kiem_tra_khoa_nb(nb, loi):
    """(3) Kỷ luật khóa tập kiểm tra trong notebook."""
    idx_unlock = None
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code" and CHUOI_MO_KHOA in cell.source:
            idx_unlock = i
            break
    if idx_unlock is None:
        loi.append(f"[3] Không tìm thấy cell mở khóa chứa '{CHUOI_MO_KHOA}'.")
        return
    for i in range(idx_unlock):
        cell = nb.cells[i]
        if cell.cell_type != "code":
            continue
        if REG_TEST.search(cell.source) and not cell.source.lstrip().startswith(THE_EDA):
            dong_dau = cell.source.splitlines()[0][:80]
            loi.append(f"[3] Cell code thứ {i} (trước cell mở khóa) chạm đường dẫn thư mục "
                       f"tập kiểm tra mà KHÔNG có thẻ '{THE_EDA}' — dòng đầu: '{dong_dau}'")


def main():
    duong_nb = Path(sys.argv[1]) if len(sys.argv) > 1 else NB_MAC_DINH
    duong_metrics = Path(sys.argv[2]) if len(sys.argv) > 2 else METRICS_MAC_DINH
    # Nếu đường dẫn tương đối không thấy từ cwd, thử lại từ gốc repo
    if not duong_nb.exists() and not duong_nb.is_absolute():
        duong_nb = GOC_REPO / duong_nb
    if not duong_metrics.exists() and not duong_metrics.is_absolute():
        duong_metrics = GOC_REPO / duong_metrics

    if not duong_nb.exists():
        print(f"LỖI: không tìm thấy notebook: {duong_nb}")
        return 1
    nb = nbformat.read(str(duong_nb), as_version=4)

    loi = []
    kiem_tra_thuc_thi(nb, loi)
    metrics = kiem_tra_metrics(duong_metrics, loi)
    kiem_tra_khoa_nb(nb, loi)

    if loi:
        print(f"VERIFY_THẤT BẠI — {len(loi)} vấn đề:")
        for l in loi[:30]:
            print("  -", l)
        if len(loi) > 30:
            print(f"  ... và {len(loi) - 30} vấn đề khác")
        return 1

    # Toàn bộ đạt → in đúng MỘT dòng VERIFY_OK (đơn vị: phần trăm, làm tròn 2 số thập phân)
    recall_theo_lop = list(metrics["per_class_recall"].values())
    ket_qua = {
        "accuracy": round(float(metrics["accuracy"]) * 100, 2),
        "macro_f1": round(float(metrics["macro_f1"]) * 100, 2),
        "min_class_recall": round(float(min(recall_theo_lop)) * 100, 2),
        "recall_spread": round(float(metrics["recall_spread"]) * 100, 2),
    }
    print("VERIFY_OK " + json.dumps(ket_qua))
    return 0


if __name__ == "__main__":
    sys.exit(main())
