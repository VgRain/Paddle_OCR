import cv2
import numpy as np
import math

# -------------------------------
# 1. Extend a detected line
# -------------------------------
def extend_line(x1, y1, x2, y2, extend_len=80):
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)

    if length == 0:
        return None

    ux = dx / length
    uy = dy / length

    sx = int(x1 - extend_len * ux)
    sy = int(y1 - extend_len * uy)
    ex = int(x2 + extend_len * ux)
    ey = int(y2 + extend_len * uy)

    return (sx, sy), (ex, ey)

# -------------------------------
# 2. Extract ROI at line ends
# -------------------------------
def extract_end_roi(img, x, y, size=35):
    h, w = img.shape[:2]
    x1 = max(0, x - size)
    y1 = max(0, y - size)
    x2 = min(w, x + size)
    y2 = min(h, y + size)
    return img[y1:y2, x1:x2]

# -------------------------------
# 3. Arrow detection
# -------------------------------
def detect_arrow(roi):
    if roi.size == 0:
        return False

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

        # Arrow head usually triangular / polygonal
        if 3 <= len(approx) <= 6:
            return True

    return False

# -------------------------------
# 4. Main pipeline
# -------------------------------
def detect_arrow_lines(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found")

    output = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Line detection (best for engineering drawings)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=50,
        maxLineGap=10
    )

    if lines is None:
        print("No lines detected")
        return output

    for line in lines:
        x1, y1, x2, y2 = line[0]

        extended = extend_line(x1, y1, x2, y2)
        if extended is None:
            continue

        start, end = extended

        roi_start = extract_end_roi(img, start[0], start[1])
        roi_end = extract_end_roi(img, end[0], end[1])

        arrow_start = detect_arrow(roi_start)
        arrow_end = detect_arrow(roi_end)

        # Classification
        if arrow_start and arrow_end:
            color = (0, 255, 0)     # Dimension line
            label = "DIMENSION"
        elif arrow_start or arrow_end:
            color = (0, 165, 255)   # Leader line
            label = "LEADER"
        else:
            color = (200, 200, 200) # Normal line
            label = "LINE"

        # Draw results
        cv2.line(output, start, end, color, 2)
        cv2.putText(
            output,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1
        )

        # Visualize ROIs (debug)
        cv2.rectangle(output,
                       (start[0]-35, start[1]-35),
                       (start[0]+35, start[1]+35),
                       (255, 0, 0), 1)

        cv2.rectangle(output,
                       (end[0]-35, end[1]-35),
                       (end[0]+35, end[1]+35),
                       (255, 0, 0), 1)

    return output

# -------------------------------
# 5. Run
# -------------------------------
if __name__ == "__main__":
    image_path = "engineering_drawing.png"  # <-- change this
    result = detect_arrow_lines(image_path)

    cv2.imshow("Arrow Line Detection", result)
    cv2.imwrite("output_detected.png", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
