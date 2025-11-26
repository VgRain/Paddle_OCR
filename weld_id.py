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
        x0, y0, x1, y1, text, block_type = b
        if block_type == 0 and text.strip():
            text_blocks.append((x0, y0, x1, y1, text.strip()))
    return text_blocks


# ============================================================
# 3. DETECT RECTANGLES USING OPENCV
# ============================================================
def detect_boxes_opencv(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)

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

            if (rx <= text_center[0] <= rx + rw and
                ry <= text_center[1] <= ry + rh):

                matches.append({
                    "text": text,
                    "text_box": (tx0, ty0, tx1, ty1),
                    "rect_box": (rx, ry, rw, rh)
                })
    return matches


# ============================================================
# 5. FILTER: ONLY KEEP VALID WELD / INTEGER BOXES
# ============================================================
def valid_weld_text(text):
    text = text.strip().upper()

    # patterns:
    # W12, W6, W80, W3 etc.
    if re.fullmatch(r"W\d+", text):
        return True

    # only number: 12, 45, 100 etc.
    if re.fullmatch(r"\d+", text):
        return True

    return False


# ============================================================
# 6. MAIN FUNCTION – SAVE MARKED PDF
# ============================================================
def detect_and_mark_in_pdf(pdf_path, output_pdf="marked_output.pdf", page_number=0):
    doc = fitz.open(pdf_path)
    page = doc[page_number]

    # ---- Step 1: text detection ----
    text_blocks = extract_text_blocks(page)

    # ---- Step 2: convert to image for OpenCV ----
    img = pdf_to_image(page)

    # ---- Step 3: detect rectangles ----
    rectangles = detect_boxes_opencv(img)

    # ---- Step 4: match text to boxes ----
    matches = match_text_to_boxes(text_blocks, rectangles)

    # ---- Step 5: filter only weld-related text ----
    filtered = []
    for m in matches:
        if valid_weld_text(m["text"]):
            filtered.append(m)

    # ---- Step 6: draw annotations on PDF ----
    for m in filtered:
        rx, ry, rw, rh = m["rect_box"]
        rect = fitz.Rect(rx, ry, rx + rw, ry + rh)

        annot = page.add_rect_annot(rect)
        annot.set_colors(stroke=(1, 0, 0))   # red for weld box
        annot.update()

    # save result
    doc.save(output_pdf)
    print("Saved:", output_pdf)

    return filtered


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    result = detect_and_mark_in_pdf("drawing.pdf")
    print("\nDetected Weld Boxes:\n")
    for r in result:
        print(r)
