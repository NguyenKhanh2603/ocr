import re

RESULTS_FILE = "evaluation_results_tesseract.txt"

# Aggregates
total_chars_gt = 0
total_errors = 0
total_lines_gt = 0
total_lines_matched = 0

with open(RESULTS_FILE, "r", encoding="utf-8") as f:
    content = f.read()

# Split by each file block
blocks = content.strip().split("=== Evaluation:")

for block in blocks:
    if "TOTAL ERROR SCORE" not in block:
        continue

    # Extract numeric values
    total_error = int(re.search(r"TOTAL ERROR SCORE:\s*(\d+)", block).group(1))
    chars_in_file = list(map(int, re.findall(r"NO\. OF CHARS IN FILE:\s*(\d+),\s*(\d+)", block)[0]))
    lines_in_file = list(map(int, re.findall(r"NO\. OF LINES IN FILE:\s*(\d+),\s*(\d+)", block)[0]))
    matched_lines = int(re.search(r"NO\. OF LINES\s*:\s*(\d+)", block).group(1))

    # Update totals
    total_errors += total_error
    total_chars_gt += chars_in_file[0]
    total_lines_gt += lines_in_file[0]
    total_lines_matched += matched_lines

# ---- Final Calculations ----
cer = total_errors / total_chars_gt if total_chars_gt else 0
accuracy = 1 - cer
line_match_rate = total_lines_matched / total_lines_gt if total_lines_gt else 0

# ---- Report ----
print("📊 Overall OCR Evaluation Summary")
print("=====================================")
print(f"Total GT Characters: {total_chars_gt}")
print(f"Total Errors:        {total_errors}")
print(f"Character Error Rate (CER): {cer*100:.2f}%")
print(f"Overall Accuracy:         {accuracy*100:.2f}%")
print(f"Line Match Rate:          {line_match_rate*100:.2f}%")