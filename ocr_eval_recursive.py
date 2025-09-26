import os
import sys
import concurrent.futures
import subprocess
from pathlib import Path

# === CONFIG ===
GROUND_TRUTH_DIR = Path("check_image_hyecho")              # path to your GT folder (files: *.txt.text)
OCR_DIR = Path("images_hyecho_result")                     # path to your OCR folder (files: *.txt)
RESULTS_FILE = Path("evaluation_results_tesseract.txt")    # where to save the results
MAX_WORKERS = 8

RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

def resolve_ocr_path_for(rel_path: Path) -> Path | None:
    """
    Try to find the matching OCR file for a GT file whose relative path is rel_path.
    1) First try exact same relative path under OCR_DIR (rare in your case).
    2) If GT endswith '.txt.text', try removing the trailing '.text' so it becomes '.txt'.
    3) As a fallback, if rel_path endswith '.txt' try adding '.text' (covers the opposite case).
    """
    # 1) exact
    p = OCR_DIR / rel_path
    if p.exists():
        return p

    # 2) GT is '*.txt.text' -> OCR is '*.txt'
    if rel_path.suffix == ".text":
        candidate = (OCR_DIR / rel_path.with_suffix(""))  # drop last '.text'
        if candidate.exists():
            return candidate

    # 3) Opposite direction (rare here): '*.txt' -> '*.txt.text'
    if rel_path.suffix == ".txt":
        candidate = OCR_DIR / rel_path.with_suffix(rel_path.suffix + ".text")
        if candidate.exists():
            return candidate

    return None

# === DISCOVER TASKS ===
tasks = []  # (rel_path:str, gt_path:Path, ocr_path:Path)

for gt_path in GROUND_TRUTH_DIR.rglob("*.txt.text"):
    rel_path = gt_path.relative_to(GROUND_TRUTH_DIR)
    ocr_path = resolve_ocr_path_for(rel_path)
    if ocr_path is None:
        print(f"⚠️ No OCR match for: {rel_path} "
              f"(tried '{rel_path}' and '{rel_path.with_suffix('')}' under {OCR_DIR})")
        continue
    tasks.append((str(rel_path), gt_path, ocr_path))

total = len(tasks)
if total == 0:
    print("❌ No evaluation pairs found. Check your directories or file extensions.")
    sys.exit(1)

print(f"🚀 Starting evaluations for {total} file pair(s) with up to {MAX_WORKERS} parallel workers...")

# === WORKER ===
def run_eval(task):
    rel_path, gt_path, ocr_path = task
    try:
        proc = subprocess.run(
            [sys.executable, "kaggle.py", str(gt_path), str(ocr_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        header = (
            f"=== Evaluation: {rel_path} ===\n"
            f"[GT ] {gt_path}\n"
            f"[OCR] {ocr_path}\n"
        )
        block = header + proc.stdout + ("\n" if not proc.stdout.endswith("\n") else "") + ("=" * 80) + "\n"
        return (rel_path, proc.returncode, block, proc.stderr)
    except Exception as e:
        header = f"=== Evaluation: {rel_path} ==="
        block = header + f"\n[EXCEPTION] {e}\n" + ("=" * 80) + "\n"
        return (rel_path, 1, block, str(e))

# === RUN PARALLEL ===
results_blocks = []
fail_count = 0

with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    future_to_task = {ex.submit(run_eval, t): t for t in tasks}
    done = 0
    for future in concurrent.futures.as_completed(future_to_task):
        rel_path, code, block, stderr = future.result()
        results_blocks.append(block)
        done += 1
        if code != 0:
            fail_count += 1
            print(f"❗ {rel_path} failed (code {code}).")
            if stderr:
                first_line = stderr.strip().splitlines()[0] if stderr.strip() else ""
                if first_line:
                    print(f"   ↳ {first_line}")
        else:
            print(f"✅ {rel_path} ({done}/{total})")

# === SAVE ALL RESULTS ===
RESULTS_FILE.write_text("".join(results_blocks), encoding="utf-8")
print("🎉 All evaluations complete!")
print(f"📊 Results saved to: {RESULTS_FILE.resolve()}")
print(f"🔎 Summary: {total - fail_count} succeeded, {fail_count} failed.")
