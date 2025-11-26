import fitz
import cv2
import numpy as np
import re


# ============================================================
# 1. CONVERT PDF PAGE TO IMAGE (for OpenCV)
# ============================================================

def pdf_to_image(page, dpi=300):
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, 3
    )
    return img


# ============================================================
# 2. GET TEXT BLOCKS FROM FITZ (NO OCR)
# ============================================================

def extract_text_blocks(page):
    blocks = page.get_text("blocks")
    text_blocks = []

    for b in blocks:
        x0, y0, x1, y1 = b[0], b[1], b[2], b[3]
        text = b[4]
        block_type = b[-1]   # type is last

        if block_type == 0 and isinstance(text, str) and text.strip():
            text_blocks.append((x0, y0, x1, y1, text.strip()))

    return text_blocks


# ============================================================
# 3. DETECT RECTANGLES USING OPENCV
# ============================================================

def detect_boxes_opencv(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    rectangles = []

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)

            # Size filters for weld boxes
            if 50 < w < 500 and 10 < h < 150 and w * h > 400:
                rectangles.append((x, y, w, h))

    return rectangles


# ============================================================
# 4. WELD TEXT FILTER
# ============================================================

def is_weld_text(text):
    text = text.strip()

    # W-prefix weld ID
    if re.fullmatch(r"[Ww]\d+[A-Za-z0-9]*", text):
        return True

    # numeric weld ID
    if re.fullmatch(r"\d+", text):
        return True

    # alphanumeric without punctuation
    if re.fullmatch(r"[A-Za-z]+\d+", text):
        return True

    return False


# ============================================================
# 5. MATCH TEXT BLOCKS TO RECTANGLES (WELD ONLY)
# ============================================================

def match_weld_boxes(text_blocks, rectangles):
    matches = []

    for (tx0, ty0, tx1, ty1, text) in text_blocks:
        text_center = ((tx0 + tx1) / 2, (ty0 + ty1) / 2)

        if not is_weld_text(text):
            continue  # skip non-weld text

        for (rx, ry, rw, rh) in rectangles:

            if (rx <= text_center[0] <= rx + rw and
                ry <= text_center[1] <= ry + rh):

                matches.append({
                    "text": text,
                    "text_box": (tx0, ty0, tx1, ty1),
                    "rect_box": (rx, ry, rw, rh)
                })

    return matches


# ============================================================
# 6. MAIN FUNCTION
# ============================================================

def detect_weld_boxes(pdf_path, page_number=0):
    doc = fitz.open(pdf_path)
    page = doc[page_number]

    print("Extracting text blocks...")
    text_blocks = extract_text_blocks(page)

    print("Rendering page...")
    img = pdf_to_image(page)

    print("Detecting rectangles...")
    rectangles = detect_boxes_opencv(img)

    print("Filtering & Matching weld boxes...")
    matches = match_weld_boxes(text_blocks, rectangles)

    # Visual output
    out = img.copy()

    for m in matches:
        rx, ry, rw, rh = m["rect_box"]
        cv2.rectangle(out, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)

        tx0, ty0, tx1, ty1 = m["text_box"]
        cv2.circle(out, (int((tx0 + tx1) / 2), int((ty0 + ty1) / 2)), 5, (0, 0, 255), -1)

        cv2.putText(out, m["text"], (rx, ry - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.imwrite("weld_boxes_detected.png", out)
    print("Saved: weld_boxes_detected.png")

    return matches


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    result = detect_weld_boxes("drawing.pdf")
    print("\nDetected Weld Boxes:\n")
    for r in result:
        print(r)# ============================================================

def detect_boxes_opencv(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    rectangles = []

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

        # only closed 4-sided shapes
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)

            # ignore tiny shapes/noise
            if w * h > 200:
                rectangles.append((x, y, w, h))

    return rectangles


# ============================================================
# 4. MATCH TEXT BLOCKS TO RECTANGLES
# ============================================================

def match_text_to_boxes(text_blocks, rectangles):
    matches = []

    for (tx0, ty0, tx1, ty1, text) in text_blocks:
        text_center = ((tx0 + tx1) / 2, (ty0 + ty1) / 2)

        for (rx, ry, rw, rh) in rectangles:

            # Check if text is inside the rectangle (with margin)
            if (rx <= text_center[0] <= rx + rw and
                ry <= text_center[1] <= ry + rh):

                matches.append({
                    "text": text,
                    "text_box": (tx0, ty0, tx1, ty1),
                    "rect_box": (rx, ry, rw, rh)
                })

    return matches


# ============================================================
# 5. MAIN FUNCTION
# ============================================================

def detect_text_boxes(pdf_path, page_number=0):
    doc = fitz.open(pdf_path)
    page = doc[page_number]

    print("Extracting text blocks...")
    text_blocks = extract_text_blocks(page)

    print("Rendering page...")
    img = pdf_to_image(page)

    print("Detecting rectangles...")
    rectangles = detect_boxes_opencv(img)

    print("Matching text to boxes...")
    matches = match_text_to_boxes(text_blocks, rectangles)

    # Draw boxes on output image
    out = img.copy()

    for m in matches:
        rx, ry, rw, rh = m["rect_box"]
        tx0, ty0, tx1, ty1 = m["text_box"]

        # rectangle around drawing box
        cv2.rectangle(out, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)

        # mark text center
        cv2.circle(out, (int((tx0 + tx1) / 2), int((ty0 + ty1) / 2)), 5, (0, 0, 255), -1)

        # put label
        cv2.putText(out, m["text"], (rx, ry - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imwrite("detected_boxes.png", out)
    print("Output saved as detected_boxes.png")

    return matches


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    result = detect_text_boxes("drawing.pdf")
    print("\nDetected boxes:\n")
    for r in result:
        print(r)
