import fitz  # PyMuPDF
import re

# Regex to detect number with degree symbol (e.g., 30°, 12.5°, 3°)
degree_pattern = re.compile(r"\b\d+(\.\d+)?°")

pdf_path = "drawing.pdf"
doc = fitz.open(pdf_path)

results = []

for page_num, page in enumerate(doc):
    text_instances = page.get_text("blocks")  # list of (x0, y0, x1, y1, text, block_no,..)
    
    for block in text_instances:
        bbox = block[:4]
        text = block[4]

        # find all occurrences inside the text block
        matches = degree_pattern.findall(text)

        # use re.finditer to get start/end positions
        for match in re.finditer(degree_pattern, text):
            matched_str = match.group()
            start = match.start()
            end = match.end()

            # get exact character-level bounding boxes
            chars = page.get_text("chars")
            char_boxes = [c for c in chars if c[4] in matched_str]

            if char_boxes:
                x0 = min(c[0] for c in char_boxes)
                y0 = min(c[1] for c in char_boxes)
                x1 = max(c[2] for c in char_boxes)
                y1 = max(c[3] for c in char_boxes)

                results.append({
                    "text": matched_str,
                    "page": page_num + 1,
                    "bbox": (x0, y0, x1, y1)
                })

# Print the results
for item in results:
    print(item)
