
# img="DLDR5BP24BK_00.jpg" # có thể thay đổi đường dẫn ảnh để test những ảnh khác
# from paddleocr import PaddleOCR
# from pathlib import Path
# import json,shutil
# ocr = PaddleOCR(
#     lang="korean",
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False)
# ROOT = Path(__file__).resolve().parent
# out_dir = ROOT / "output"
# out_dir.mkdir(parents=True, exist_ok=True)

# # Run OCR inference on a sample image
# r = ocr.predict(img)
# for l in r:
#     l.save_to_json(str(out_dir))

# # print(r)
# json_path = out_dir / (Path(img).stem + "_res.json")
# with open(json_path, "r", encoding="utf-8") as f:
#     obj = json.load(f)
# from statistics import median

# def _normalize_poly(poly):
#     """Trả về 4 điểm [[x,y],...], hoặc None nếu không hợp lệ."""
#     if poly is None:
#         return None
#     # poly dạng [[x,y],...]
#     if isinstance(poly, (list, tuple)) and len(poly) >= 4 and all(
#         isinstance(p, (list, tuple)) and len(p) >= 2 for p in poly
#     ):
#         return [[int(p[0]), int(p[1])] for p in poly[:4]]
#     return None

# def _poly_from_box(box):
#     """rec_boxes: [x1,y1,x2,y2] -> 4 đỉnh theo thứ tự."""
#     if not (isinstance(box, (list, tuple)) and len(box) == 4):
#         return None
#     x1, y1, x2, y2 = map(int, box)
#     return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

# def _pick_page_dict(result):
#     """predict() có thể trả dict hoặc list[dict]; lấy dict đầu."""
#     if isinstance(result, dict):
#         return result
#     if isinstance(result, (list, tuple)) and result and isinstance(result[0], dict):
#         return result[0]
#     raise ValueError("Không nhận ra cấu trúc result; cần dict hoặc list[dict].")

# def print_reading_order(result, y_merge=0.6, gap_mult=1.8):
#     page = _pick_page_dict(result)

#     # Lấy text & scores
#     texts  = page.get("rec_texts") or []
#     scores = page.get("rec_scores") or [None]*len(texts)

#     # Lấy polygon: ưu tiên rec_polys; nếu không có thì dùng rec_boxes
#     polys = page.get("rec_polys")
#     if polys:
#         polys = [ _normalize_poly(p) for p in polys ]
#     else:
#         boxes = page.get("rec_boxes") or []
#         polys = [ _poly_from_box(b) for b in boxes ]

#     # Cắt theo độ dài min, bỏ item thiếu polygon/text
#     n = min(len(texts), len(scores), len(polys))
#     items = []
#     for i in range(n):
#         poly = polys[i]
#         txt  = texts[i]
#         if not poly or not txt:
#             continue
#         xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
#         top  = min(ys)
#         left = min(xs)
#         h    = max(1, max(ys) - min(ys))
#         items.append({"txt": txt, "top": top, "left": left, "h": h})

#     if not items:
#         print("")  # không có gì để in
#         return

#     # Sort toàn cục theo (top, left)
#     items.sort(key=lambda x: (x["top"], x["left"]))

#     # Tham số nhóm dòng theo trục y
#     med_h = median([it["h"] for it in items]) if items else 16
#     y_thr = max(4, med_h * y_merge)

#     # Gom thành các dòng
#     rows, cur = [], []
#     for it in items:
#         if not cur:
#             cur = [it]
#             continue
#         cur_mean_top = sum(c["top"] for c in cur) / len(cur)
#         if abs(it["top"] - cur_mean_top) <= y_thr:
#             cur.append(it)
#         else:
#             rows.append(cur); cur = [it]
#     if cur: rows.append(cur)

#     # In: trong từng dòng sắp xếp trái→phải; chèn dòng trống nếu cách xa theo trục y
#     prev_mid, out_lines = None, []
#     for r in rows:
#         r.sort(key=lambda x: x["left"])
#         mid = sum(x["top"] for x in r) / len(r)
#         if prev_mid is not None and (mid - prev_mid) > (gap_mult * med_h):
#             out_lines.append("")  # ngắt đoạn khi khoảng cách dọc lớn
#         out_lines.append(" ".join(x["txt"] for x in r))
#         prev_mid = mid

#     print("\n".join(out_lines))

# # ==== GỌI HÀM ====
# print_reading_order(obj, y_merge=0.6, gap_mult=1.8)
# shutil.rmtree(out_dir, ignore_errors=True)


from pathlib import Path
import argparse
import json
import sys
from statistics import median
from typing import List
import shutil
from PIL import Image

from paddleocr import PaddleOCR

ROOT = Path(__file__).resolve().parent
OUT_JSON_DIR = ROOT / "output"
OUT_JSON_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp",
    ".tif",
    ".tiff",
}


def _normalize_poly(poly):
    """Return first 4 points [[x, y], ...] or None when invalid."""
    if poly is None:
        return None
    if isinstance(poly, (list, tuple)) and len(poly) >= 4:
        points = []
        for p in poly:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                return None
            points.append([int(p[0]), int(p[1])])
            if len(points) == 4:
                return points
    return None


def _poly_from_box(box):
    """Convert rec_boxes [x1, y1, x2, y2] into 4-point polygon."""
    if not (isinstance(box, (list, tuple)) and len(box) == 4):
        return None
    x1, y1, x2, y2 = map(int, box)
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _pick_page_dict(result):
    """Ensure we always work with a dict representing a single OCR page."""
    if isinstance(result, dict):
        return result
    if isinstance(result, (list, tuple)) and result and isinstance(result[0], dict):
        return result[0]
    raise ValueError("Unrecognized OCR result structure; expected dict or list[dict].")


def reading_order_text(result, y_merge=0.6, gap_mult=1.8):
    page = _pick_page_dict(result)

    texts = page.get("rec_texts") or []
    polys = page.get("rec_polys")
    if polys:
        polys = [_normalize_poly(p) for p in polys]
    else:
        boxes = page.get("rec_boxes") or []
        polys = [_poly_from_box(b) for b in boxes]

    n = min(len(texts), len(polys))
    items = []
    for i in range(n):
        poly = polys[i]
        txt = texts[i]
        if not poly or not txt:
            continue
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        top = min(ys)
        left = min(xs)
        h = max(1, max(ys) - min(ys))
        items.append({"txt": txt, "top": top, "left": left, "h": h})

    if not items:
        return ""

    items.sort(key=lambda x: (x["top"], x["left"]))

    med_h = median([it["h"] for it in items]) if items else 16
    y_thr = max(4, med_h * y_merge)

    rows, cur = [], []
    for it in items:
        if not cur:
            cur = [it]
            continue
        cur_mean_top = sum(c["top"] for c in cur) / len(cur)
        if abs(it["top"] - cur_mean_top) <= y_thr:
            cur.append(it)
        else:
            rows.append(cur)
            cur = [it]
    if cur:
        rows.append(cur)

    prev_mid = None
    out_lines = []
    for r in rows:
        r.sort(key=lambda x: x["left"])
        mid = sum(x["top"] for x in r) / len(r)
        if prev_mid is not None and (mid - prev_mid) > (gap_mult * med_h):
            out_lines.append("")
        out_lines.append(" ".join(x["txt"] for x in r))
        prev_mid = mid

    return "\n".join(out_lines)


def _is_valid_image(image_path: Path) -> bool:
    """Check if an image file is valid and can be opened."""
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify it's a valid image
        return True
    except Exception as e:
        print(f"Invalid image {image_path.name}: {e}", file=sys.stderr)
        return False


def _convert_and_enhance_image(image_path: Path) -> Path:
    """Convert AVIF images to JPEG and upscale small images for better OCR."""
    try:
        with Image.open(image_path) as img:
            needs_processing = False
            temp_path = image_path
            
            # Check if conversion is needed (AVIF or small size)
            if img.format == 'AVIF':
                needs_processing = True
                print(f"Converting AVIF image {image_path.name} to JPEG for processing...", file=sys.stderr)
            
            # Check if upscaling is needed (images smaller than 500px in any dimension)
            min_size = 500
            if img.size[0] < min_size or img.size[1] < min_size:
                needs_processing = True
                scale_factor = max(min_size / img.size[0], min_size / img.size[1])
                new_size = (int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))
                print(f"Upscaling small image {image_path.name} from {img.size} to {new_size} for better OCR...", file=sys.stderr)
            
            if needs_processing:
                # Create a temporary JPEG file
                temp_path = image_path.parent / f"temp_enhanced_{image_path.stem}.jpg"
                
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                
                # Upscale if needed
                if img.size[0] < min_size or img.size[1] < min_size:
                    # Use LANCZOS for high-quality upscaling
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                img.save(temp_path, 'JPEG', quality=95)
                return temp_path
                
        return image_path
    except Exception as e:
        print(f"Error processing {image_path.name}: {e}", file=sys.stderr)
        return image_path


def _collect_images(path: Path) -> List[Path]:
    """Return list of valid image paths under the given path."""
    if path.is_dir():
        all_images = [p for p in sorted(path.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        images = [img for img in all_images if _is_valid_image(img)]
    elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        if _is_valid_image(path):
            images = [path]
        else:
            images = []
    else:
        images = []
    return images


def process_image(ocr: PaddleOCR, image_path: Path, keep_json: bool, y_merge: float, gap_mult: float) -> str:
    """Run OCR for a single image path and return extracted text."""
    temp_path = None
    try:
        # Convert AVIF to JPEG and enhance image if needed
        processing_path = _convert_and_enhance_image(image_path)
        if processing_path != image_path:
            temp_path = processing_path
        
        result = ocr.predict(str(processing_path))
        if not isinstance(result, (list, tuple)):
            result = [result]
        for page in result:
            page.save_to_json(str(OUT_JSON_DIR))

        # Use the actual processed filename for JSON output (not the original)
        json_path = OUT_JSON_DIR / f"{processing_path.stem}_res.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        text = reading_order_text(data, y_merge=y_merge, gap_mult=gap_mult)

        if not keep_json:
            json_path.unlink(missing_ok=True)

        return text
    except Exception as e:
        print(f"Error processing {image_path.name}: {e}", file=sys.stderr)
        return ""
    finally:
        # Clean up temporary file
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Run PaddleOCR on an image file or every image in a folder.")
    parser.add_argument("input_path", help="Path to an image file or directory containing images.")
    parser.add_argument("--outdir", default=str(ROOT / "results"), help="Directory to save extracted text files.")
    parser.add_argument("--keep-json", action="store_true", help="Keep intermediate PaddleOCR JSON outputs.")
    parser.add_argument("--y-merge", type=float, default=0.6, help="Y merge factor for grouping text lines.")
    parser.add_argument("--gap-mult", type=float, default=1.8, help="Gap multiplier to detect paragraph breaks.")
    parser.add_argument("--lang", default="korean", help="PaddleOCR language model to use.")
    args = parser.parse_args()

    input_path = Path(args.input_path).resolve()
    if not input_path.exists():
        print(f"Input path not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    images = _collect_images(input_path)
    if not images:
        print(f"No supported image files found under {input_path}", file=sys.stderr)
        sys.exit(1)

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    ocr = PaddleOCR(
        lang=args.lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    print(f"Processing {len(images)} image(s) from {input_path}...", file=sys.stderr)

    for image_path in images:
        print(f"Processing {image_path.name}...", file=sys.stderr)
        text = process_image(ocr, image_path, args.keep_json, args.y_merge, args.gap_mult)
        print(f"\n===== {image_path.name} =====")
        if text:
            print(text)
        else:
            print("(No text extracted or processing failed)")
        
        output_file = outdir / f"{image_path.stem}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved to: {output_file}", file=sys.stderr)

    if not args.keep_json and OUT_JSON_DIR.exists():
        try:
            next(OUT_JSON_DIR.iterdir())
        except StopIteration:
            shutil.rmtree(OUT_JSON_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()

