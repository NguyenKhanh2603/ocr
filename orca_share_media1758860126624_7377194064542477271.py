#
# img="TAF20253_00.jpg"
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
# # image=ROOT/"images_hyecho"
# out_dir.mkdir(parents=True, exist_ok=True)
#
# # Run OCR inference on a sample image
# r = ocr.predict(img)
# for l in r:
#     l.save_to_json(str(out_dir))
#
# # print(r)
# json_path = out_dir / (Path(img).stem + "_res.json")
# with open(json_path, "r", encoding="utf-8") as f:
#     obj = json.load(f)
# from statistics import median
#
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
#
# def _poly_from_box(box):
#     """rec_boxes: [x1,y1,x2,y2] -> 4 đỉnh theo thứ tự."""
#     if not (isinstance(box, (list, tuple)) and len(box) == 4):
#         return None
#     x1, y1, x2, y2 = map(int, box)
#     return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
#
# def _pick_page_dict(result):
#     """predict() có thể trả dict hoặc list[dict]; lấy dict đầu."""
#     if isinstance(result, dict):
#         return result
#     if isinstance(result, (list, tuple)) and result and isinstance(result[0], dict):
#         return result[0]
#     raise ValueError("Không nhận ra cấu trúc result; cần dict hoặc list[dict].")
#
# def print_reading_order(result, y_merge=0.6, gap_mult=1.8):
#     page = _pick_page_dict(result)
#
#     # Lấy text & scores
#     texts  = page.get("rec_texts") or []
#     scores = page.get("rec_scores") or [None]*len(texts)
#
#     # Lấy polygon: ưu tiên rec_polys; nếu không có thì dùng rec_boxes
#     polys = page.get("rec_polys")
#     if polys:
#         polys = [ _normalize_poly(p) for p in polys ]
#     else:
#         boxes = page.get("rec_boxes") or []
#         polys = [ _poly_from_box(b) for b in boxes ]
#
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
#
#     if not items:
#         print("")  # không có gì để in
#         return
#
#     # Sort toàn cục theo (top, left)
#     items.sort(key=lambda x: (x["top"], x["left"]))
#
#     # Tham số nhóm dòng theo trục y
#     med_h = median([it["h"] for it in items]) if items else 16
#     y_thr = max(4, med_h * y_merge)
#
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
#
#     # In: trong từng dòng sắp xếp trái→phải; chèn dòng trống nếu cách xa theo trục y
#     prev_mid, out_lines = None, []
#     for r in rows:
#         r.sort(key=lambda x: x["left"])
#         mid = sum(x["top"] for x in r) / len(r)
#         if prev_mid is not None and (mid - prev_mid) > (gap_mult * med_h):
#             out_lines.append("")  # ngắt đoạn khi khoảng cách dọc lớn
#         out_lines.append(" ".join(x["txt"] for x in r))
#         prev_mid = mid
#
#     print("\n".join(out_lines))
#
# # ==== GỌI HÀM ====
# print_reading_order(obj, y_merge=0.6, gap_mult=1.8)
# shutil.rmtree(out_dir, ignore_errors=True)
# # # run_ocr_batch.py
# # from pathlib import Path
# # import os, json, shutil, traceback
# # from concurrent.futures import ProcessPoolExecutor, as_completed
# # from statistics import median
# #
# # # ---------- READING-ORDER HELPERS ----------
# #
# # def _normalize_poly(poly):
# #     if poly is None:
# #         return None
# #     if hasattr(poly, "tolist"):
# #         poly = poly.tolist()
# #     if isinstance(poly, (list, tuple)) and len(poly) >= 4 and all(
# #         isinstance(p, (list, tuple)) and len(p) >= 2 for p in poly
# #     ):
# #         return [[int(p[0]), int(p[1])] for p in poly[:4]]
# #     if isinstance(poly, (list, tuple)) and len(poly) == 4:
# #         x1, y1, x2, y2 = map(int, poly)
# #         return [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
# #     return None
# #
# # def _reading_order_from_page_dict(page, y_merge=0.6, gap_mult=1.8):
# #     texts  = page.get("rec_texts")  or []
# #     scores = page.get("rec_scores") or [None]*len(texts)
# #     polys  = page.get("rec_polys")
# #     if polys:
# #         polys = [ _normalize_poly(p) for p in polys ]
# #     else:
# #         boxes = page.get("rec_boxes") or []
# #         polys = [ _normalize_poly(b) for b in boxes ]
# #
# #     n = min(len(texts), len(scores), len(polys))
# #     items = []
# #     for i in range(n):
# #         poly = polys[i]; txt = texts[i]
# #         if not poly or not txt:
# #             continue
# #         xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
# #         items.append({
# #             "txt": str(txt),
# #             "top": min(ys),
# #             "left": min(xs),
# #             "h": max(1, max(ys)-min(ys))
# #         })
# #     if not items:
# #         return []
# #
# #     items.sort(key=lambda x: (x["top"], x["left"]))
# #     med_h = median([it["h"] for it in items])
# #     y_thr = max(4.0, med_h * y_merge)
# #
# #     rows, cur = [], []
# #     for it in items:
# #         if not cur:
# #             cur = [it]; continue
# #         mean_top = sum(c["top"] for c in cur) / len(cur)
# #         if abs(it["top"] - mean_top) <= y_thr:
# #             cur.append(it)
# #         else:
# #             rows.append(cur); cur = [it]
# #     if cur: rows.append(cur)
# #
# #     out, prev_mid = [], None
# #     for r in rows:
# #         r.sort(key=lambda x: x["left"])
# #         mid = sum(x["top"] for x in r) / len(r)
# #         if prev_mid is not None and (mid - prev_mid) > (gap_mult * med_h):
# #             out.append("")  # paragraph break
# #         out.append(" ".join(x["txt"] for x in r))
# #         prev_mid = mid
# #     return out
# #
# # # ---------- PER-IMAGE WORKER ----------
# #
# # # We'll lazily create a PaddleOCR instance per process:
# # _OCR = None
# # def _get_ocr(lang="korean"):
# #     global _OCR
# #     if _OCR is None:
# #         from paddleocr import PaddleOCR
# #         _OCR = PaddleOCR(
# #             lang=lang,
# #             use_doc_orientation_classify=False,
# #             use_doc_unwarping=False,
# #             use_textline_orientation=False
# #
# #         )
# #     return _OCR
# #
# # def process_one_image(img_path:str, tmp_dir:str, out_txt_dir:str, lang:str="korean") -> str:
# #     """
# #     Runs OCR on one image, saves a JSON to tmp_dir, extracts text in reading order,
# #     writes <stem>.txt to out_txt_dir, and removes the JSON.
# #     Returns the path of the .txt written (or raises on error).
# #     """
# #     img = Path(img_path)
# #     tmp_dir = Path(tmp_dir)
# #     out_txt_dir = Path(out_txt_dir)
# #
# #     ocr = _get_ocr(lang=lang)
# #     res = ocr.predict(str(img))
# #     # save JSON(s) – predict can return list of pages
# #     pages = res if isinstance(res, (list, tuple)) else [res]
# #     for p in pages:
# #         p.save_to_json(str(tmp_dir))
# #
# #     json_path = tmp_dir / f"{img.stem}_res.json"
# #     if not json_path.exists():
# #         # In some versions the filename may differ; fallback to first *_res.json for this stem
# #         candidates = list(tmp_dir.glob(f"{img.stem}*_res.json"))
# #         if candidates:
# #             json_path = candidates[0]
# #         else:
# #             raise FileNotFoundError(f"No JSON found for {img.name}")
# #
# #     with open(json_path, "r", encoding="utf-8") as f:
# #         page_dict = json.load(f)
# #
# #     lines = _reading_order_from_page_dict(page_dict)
# #
# #     out_txt = out_txt_dir / f"{img.stem}.txt"
# #     out_txt.write_text("\n".join(lines), encoding="utf-8")
# #
# #     # clean the single JSON file (keep tmp_dir for others)
# #     try:
# #         json_path.unlink(missing_ok=True)
# #     except Exception:
# #         pass
# #
# #     return str(out_txt)
# #
# # # ---------- BATCH DRIVER ----------
# #
# # def main():
# #     ROOT = Path(__file__).resolve().parent
# #     src_dir = ROOT / "images_lfmall1"           # folder containing many images
# #     tmp_dir = ROOT / "output_tmp_json"         # per-image JSONs (temporary)
# #     dst_dir = ROOT / "images_lfmall1_result"    # where .txt files go
# #
# #     # Create dirs
# #     tmp_dir.mkdir(parents=True, exist_ok=True)
# #     dst_dir.mkdir(parents=True, exist_ok=True)
# #
# #     # Collect images
# #     exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# #     images = [p for p in src_dir.glob("*") if p.suffix.lower() in exts]
# #     if not images:
# #         print(f"No images found in: {src_dir}")
# #         return
# #
# #     # Parallel run
# #     max_workers = max(1, os.cpu_count() - 1)  # leave 1 core free
# #     print(f"Processing {len(images)} images with {max_workers} workers...")
# #
# #     futures = []
# #     with ProcessPoolExecutor(max_workers=max_workers) as ex:
# #         for img in images:
# #             futures.append(
# #                 ex.submit(process_one_image, str(img), str(tmp_dir), str(dst_dir), "korean")
# #             )
# #
# #         ok = 0
# #         for fut in as_completed(futures):
# #             try:
# #                 txt_path = fut.result()
# #                 ok += 1
# #                 print(f"OK: {txt_path}")
# #             except Exception as e:
# #                 print("ERROR:", e)
# #                 traceback.print_exc()
# #
# #     # Try to remove tmp folder if empty
# #     try:
# #         if tmp_dir.exists():
# #             # if it's empty, remove directly; else keep (maybe concurrent files remained)
# #             if not any(tmp_dir.iterdir()):
# #                 tmp_dir.rmdir()
# #             else:
# #                 # you can force delete everything if you prefer:
# #                 # shutil.rmtree(tmp_dir, ignore_errors=True)
# #                 pass
# #     except Exception:
# #         pass
# #
# #     print(f"Done. Wrote {ok} txt files to: {dst_dir}")
# #
# # if __name__ == "__main__":
# #     main()
# # run_ocr_batch_threads.py

# run_ocr_batch_tempjson.py
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import median
import os, json, shutil, traceback

# ---------- reading-order helpers ----------
def _normalize_poly(poly):
    if poly is None: return None
    if hasattr(poly, "tolist"): poly = poly.tolist()
    if isinstance(poly, (list, tuple)) and len(poly) >= 4 and all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in poly):
        return [[int(p[0]), int(p[1])] for p in poly[:4]]
    if isinstance(poly, (list, tuple)) and len(poly) == 4:
        x1,y1,x2,y2 = map(int, poly)
        return [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
    return None

def _reading_order_from_page_dict(page, y_merge=0.6, gap_mult=1.8):
    texts  = page.get("rec_texts")  or []
    scores = page.get("rec_scores") or [None]*len(texts)
    polys  = page.get("rec_polys")
    if polys:
        polys = [ _normalize_poly(p) for p in polys ]
    else:
        boxes = page.get("rec_boxes") or []
        polys = [ _normalize_poly(b) for b in boxes ]

    n = min(len(texts), len(scores), len(polys))
    items = []
    for i in range(n):
        poly, txt = polys[i], texts[i]
        if not poly or not txt:
            continue
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        items.append({"txt": str(txt), "top": min(ys), "left": min(xs), "h": max(1, max(ys)-min(ys))})
    if not items:
        return []

    items.sort(key=lambda x: (x["top"], x["left"]))
    med_h = median([it["h"] for it in items])
    y_thr = max(4.0, med_h * y_merge)

    rows, cur = [], []
    for it in items:
        if not cur:
            cur = [it]; continue
        mean_top = sum(c["top"] for c in cur)/len(cur)
        if abs(it["top"] - mean_top) <= y_thr:
            cur.append(it)
        else:
            rows.append(cur); cur = [it]
    if cur: rows.append(cur)

    out, prev_mid = [], None
    for r in rows:
        r.sort(key=lambda x: x["left"])
        mid = sum(x["top"] for x in r)/len(r)
        if prev_mid is not None and (mid - prev_mid) > (gap_mult * med_h):
            out.append("")  # paragraph break
        out.append(" ".join(x["txt"] for x in r))
        prev_mid = mid
    return out

# ---------- per-image task (thread-friendly) ----------
def process_one_image_thread(img_path:str, tmp_json_dir:str, out_txt_dir:str, lang:str="korean"):
    """
    - Runs OCR on one image
    - Writes JSON to tmp_json_dir/<stem>_res.json (via save_to_json)
    - Reads JSON, extracts text in reading order
    - Writes out_txt_dir/<stem>.txt
    """
    from paddleocr import PaddleOCR  # import inside thread
    ocr = PaddleOCR(
        lang=lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False

    )

    img = Path(img_path)
    tmp = Path(tmp_json_dir)
    out = Path(out_txt_dir)

    res = ocr.predict(str(img))
    pages = res if isinstance(res, (list, tuple)) else [res]
    for p in pages:
        p.save_to_json(str(tmp))  # creates <stem>_res.json

    json_path = tmp / f"{img.stem}_res.json"
    if not json_path.exists():
        # fallback if naming differs
        cands = list(tmp.glob(f"{img.stem}*_res.json"))
        if cands:
            json_path = cands[0]
        else:
            raise FileNotFoundError(f"No JSON found for {img.name}")

    with open(json_path, "r", encoding="utf-8") as f:
        page_dict = json.load(f)

    lines = _reading_order_from_page_dict(page_dict)
    out_txt_path = out / f"{img.stem}.txt"
    out_txt_path.write_text("\n".join(lines), encoding="utf-8")

    return str(out_txt_path)

def main():
    # project-root-relative paths
    ROOT = Path(__file__).resolve().parent
    src_dir   = ROOT / "images_hyecho"         # input images
    tmp_dir   = ROOT / "temp_json"             # <- the temp JSON folder you asked for
    result_dir = ROOT / "images_hyecho_result" # output .txt

    tmp_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    # collect images
    exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}
    images = [p for p in src_dir.glob("*") if p.suffix.lower() in exts]
    if not images:
        print(f"No images found in {src_dir}")
        return

    # limit threads to keep things stable
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    max_workers = os.cpu_count()-4
    print(f"Processing {len(images)} images with {max_workers} threads...")

    ok = 0
    errs = 0
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(process_one_image_thread, str(img), str(tmp_dir), str(result_dir), "korean"): img for img in images}
        for fut in as_completed(futs):
            img = futs[fut]
            try:
                txt_path = fut.result()
                ok += 1
                print("OK:", txt_path)
            except Exception as e:
                errs += 1
                print("ERROR:", img.name, "-", e)
                traceback.print_exc()

    # clean the whole temp_json folder at the end
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"Temp folder removed: {tmp_dir}")
    except Exception as e:
        print("Could not remove temp_json:", e)

    print(f"Done. {ok} succeeded, {errs} failed. Results → {result_dir}")

if __name__ == "__main__":
    main()

