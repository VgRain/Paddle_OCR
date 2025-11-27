import fitz
import math
import re


def is_small_circle(path):
    """Check whether a path is a tiny circle used as a degree symbol."""
    if "c" not in path.get("items", {}):  
        return False

    # circle approx via 4 cubic bezier curves (most CAD PDFs)
    rect = path["rect"]
    w = rect[2] - rect[0]
    h = rect[3] - rect[1]

    # tiny circle: ~1–5 px typically
    return 0.5 < w < 8 and 0.5 < h < 8


pdf = "drawing.pdf"
doc = fitz.open(pdf)

results = []

for pno, page in enumerate(doc):
    # 1) extract all paths
    paths = page.get_drawings()

    # find all small circles (degree symbols)
    degree_symbols = []
    for p in paths:
        if is_small_circle(p):
            degree_symbols.append(p["rect"])

    # 2) extract text (chars)
    chars = page.get_text("chars")

    # detect numbers near circles
    for circle in degree_symbols:
        cx = (circle[0] + circle[2]) / 2
        cy = (circle[1] + circle[3]) / 2

        # find nearby numbers (within 40px)
        nearby_chars = [
            c for c in chars
            if abs(c[1] - cy) < 20 and abs(c[0] - cx) < 40 and c[4].isdigit()
        ]

        if not nearby_chars:
            continue

        # group them to form number (e.g., '4','5' => "45")
        x_sorted = sorted(nearby_chars, key=lambda c: c[0])
        number = "".join(c[4] for c in x_sorted)

        # return bounding boxes
        num_bbox = (
            min(c[0] for c in x_sorted),
            min(c[1] for c in x_sorted),
            max(c[2] for c in x_sorted),
            max(c[3] for c in x_sorted),
        )

        results.append({
            "text": number + "°",
            "page": pno + 1,
            "circle_bbox": circle,
            "number_bbox": num_bbox
        })

print(results)
