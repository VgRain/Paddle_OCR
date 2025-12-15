import cv2
import numpy as np
import math

# -----------------------------------
# Utility: Extend a line
# -----------------------------------
def extend_line(x1, y1, x2, y2, extend=80):
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return None

    ux, uy = dx / length, dy / length
    sx, sy = int(x1 - ux * extend), int(y1 - uy * extend)
    ex, ey = int(x2 + ux * extend), int(y2 + uy * extend)
    return (sx, sy), (ex, ey)

# -----------------------------------
# Utility: Extract ROI
# -----------------------------------
def extract_roi(img, x, y, size=35):
    h, w = img.shape[:2]
    return img[max(0,y-size):min(h,y+size),
               max(0,x-size):min(w,x+size)]

# -----------------------------------
# Utility: Remove line near arrow
# -----------------------------------
def suppress_line(mask, x, y, r=15):
    cv2.circle(mask, (x, y), r, 0, -1)

# -----------------------------------
# Arrow detection (triangle-based)
# -----------------------------------
def detect_arrow(roi):
    if roi.size == 0:
        return False

    contours, _ = cv2.findContours(
        roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 80:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

        # Engineering arrow head ≈ triangle
        if len(approx) == 3:
            return True

    return False

# -----------------------------------
# Remove overlapping / duplicate lines
# -----------------------------------
def remove_duplicates(lines, tol=10):
    filtered = []
    for l in lines:
        x1,y1,x2,y2 = l
        duplicate = False
        for f in filtered:
            if (abs(x1-f[0]) < tol and abs(y1-f[1]) < tol and
                abs(x2-f[2]) < tol and abs(y2-f[3]) < tol):
                duplicate = True
                break
        if not duplicate:
            filtered.append(l)
    return filtered

# -----------------------------------
# MAIN PIPELINE
# -----------------------------------
def process_engineering_drawing(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found")

    out = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Binary image ---
    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # --- TEXT REMOVAL (CRITICAL) ---
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))

    text_removed = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
    text_removed |= cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)

    # --- Line detection ---
    edges = cv2.Canny(text_removed, 50, 150)

    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=120,
        minLineLength=120,
        maxLineGap=5
    )

    if raw_lines is None:
        return out

    lines = [l[0] for l in raw_lines]
    lines = remove_duplicates(lines)

    # --- Arrow mask (copy of binary) ---
    arrow_mask = bw.copy()

    results = []

    # --- Process each line ---
    for x1,y1,x2,y2 in lines:
        extended = extend_line(x1,y1,x2,y2)
        if not extended:
            continue

        start, end = extended

        # Suppress line near endpoints (KEY FIX)
        suppress_line(arrow_mask, start[0], start[1])
        suppress_line(arrow_mask, end[0], end[1])

        roi_start = extract_roi(arrow_mask, start[0], start[1])
        roi_end   = extract_roi(arrow_mask, end[0], end[1])

        arrow_s = detect_arrow(roi_start)
        arrow_e = detect_arrow(roi_end)

        # Classification
        if arrow_s and arrow_e:
            label, color = "DIMENSION", (0,255,0)
        elif arrow_s or arrow_e:
            label, color = "LEADER", (0,165,255)
        else:
            label, color = "LINE", (200,200,200)

        results.append((start, end, label, color))

    # --- Draw results ---
    for start, end, label, color in results:
        cv2.line(out, start, end, color, 2)
        cv2.putText(out, label, (start[0], start[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return out

# -----------------------------------
# RUN
# -----------------------------------
if __name__ == "__main__":
    img_path = "engineering_drawing.png"  # change path
    result = process_engineering_drawing(img_path)

    cv2.imshow("Final Result", result)
    cv2.imwrite("final_output.png", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
