
img="DLDR5BP24BK_00.jpg" # có thể thay đổi đường dẫn ảnh để test những ảnh khác
from paddleocr import PaddleOCR
from pathlib import Path
import json,shutil
ocr = PaddleOCR(
    lang="korean",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)
ROOT = Path(__file__).resolve().parent
out_dir = ROOT / "output"
out_dir.mkdir(parents=True, exist_ok=True)

# Run OCR inference on a sample image
r = ocr.predict(img)
for l in r:
    l.save_to_json(str(out_dir))

# print(r)
json_path = out_dir / (Path(img).stem + "_res.json")
with open(json_path, "r", encoding="utf-8") as f:
    obj = json.load(f)
from statistics import median

def _normalize_poly(poly):
    """Trả về 4 điểm [[x,y],...], hoặc None nếu không hợp lệ."""
    if poly is None:
        return None
    # poly dạng [[x,y],...]
    if isinstance(poly, (list, tuple)) and len(poly) >= 4 and all(
        isinstance(p, (list, tuple)) and len(p) >= 2 for p in poly
    ):
        return [[int(p[0]), int(p[1])] for p in poly[:4]]
    return None

def _poly_from_box(box):
    """rec_boxes: [x1,y1,x2,y2] -> 4 đỉnh theo thứ tự."""
    if not (isinstance(box, (list, tuple)) and len(box) == 4):
        return None
    x1, y1, x2, y2 = map(int, box)
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

def _pick_page_dict(result):
    """predict() có thể trả dict hoặc list[dict]; lấy dict đầu."""
    if isinstance(result, dict):
        return result
    if isinstance(result, (list, tuple)) and result and isinstance(result[0], dict):
        return result[0]
    raise ValueError("Không nhận ra cấu trúc result; cần dict hoặc list[dict].")

def print_reading_order(result, y_merge=0.6, gap_mult=1.8):
    page = _pick_page_dict(result)

    # Lấy text & scores
    texts  = page.get("rec_texts") or []
    scores = page.get("rec_scores") or [None]*len(texts)

    # Lấy polygon: ưu tiên rec_polys; nếu không có thì dùng rec_boxes
    polys = page.get("rec_polys")
    if polys:
        polys = [ _normalize_poly(p) for p in polys ]
    else:
        boxes = page.get("rec_boxes") or []
        polys = [ _poly_from_box(b) for b in boxes ]

    # Cắt theo độ dài min, bỏ item thiếu polygon/text
    n = min(len(texts), len(scores), len(polys))
    items = []
    for i in range(n):
        poly = polys[i]
        txt  = texts[i]
        if not poly or not txt:
            continue
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        top  = min(ys)
        left = min(xs)
        h    = max(1, max(ys) - min(ys))
        items.append({"txt": txt, "top": top, "left": left, "h": h})

    if not items:
        print("")  # không có gì để in
        return

    # Sort toàn cục theo (top, left)
    items.sort(key=lambda x: (x["top"], x["left"]))

    # Tham số nhóm dòng theo trục y
    med_h = median([it["h"] for it in items]) if items else 16
    y_thr = max(4, med_h * y_merge)

    # Gom thành các dòng
    rows, cur = [], []
    for it in items:
        if not cur:
            cur = [it]
            continue
        cur_mean_top = sum(c["top"] for c in cur) / len(cur)
        if abs(it["top"] - cur_mean_top) <= y_thr:
            cur.append(it)
        else:
            rows.append(cur); cur = [it]
    if cur: rows.append(cur)

    # In: trong từng dòng sắp xếp trái→phải; chèn dòng trống nếu cách xa theo trục y
    prev_mid, out_lines = None, []
    for r in rows:
        r.sort(key=lambda x: x["left"])
        mid = sum(x["top"] for x in r) / len(r)
        if prev_mid is not None and (mid - prev_mid) > (gap_mult * med_h):
            out_lines.append("")  # ngắt đoạn khi khoảng cách dọc lớn
        out_lines.append(" ".join(x["txt"] for x in r))
        prev_mid = mid

    print("\n".join(out_lines))

# ==== GỌI HÀM ====
print_reading_order(obj, y_merge=0.6, gap_mult=1.8)
shutil.rmtree(out_dir, ignore_errors=True)

