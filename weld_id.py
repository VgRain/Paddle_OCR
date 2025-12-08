import cv2
import numpy as np

def detect_arrows_and_lines(img_path):

    img = cv2.imread(img_path)
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binary threshold
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ---- STEP 1: FIND CONTOURS (Arrow Candidates) ----
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    arrow_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:  # ignore noise
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        vertices = len(approx)

        # Compute circularity
        if peri == 0:
            circularity = 1
        else:
            circularity = 4 * np.pi * area / (peri * peri)

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)

        # ---- Arrow Condition ----
        if (5 <= vertices <= 8) and (circularity < 0.6) and (aspect_ratio < 0.75 or aspect_ratio > 1.3):
            arrow_contours.append(cnt)
            cv2.drawContours(img, [cnt], -1, (0, 255, 0), 3)   # green = arrow

    # ---- STEP 2: DETECT LINES (Hough Transform) ----

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                            minLineLength=50, maxLineGap=10)

    detected_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            detected_lines.append((x1, y1, x2, y2))
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # blue = line

    # ---- STEP 3: CHECK IF LINE PASSES THROUGH ARROW ----

    for cnt in arrow_contours:

        # Convert contour to mask
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)

        for (x1, y1, x2, y2) in detected_lines:

            # Draw single-pixel line on mask
            line_mask = np.zeros_like(mask)
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 1)

            # Intersection = bitwise AND
            intersection = cv2.bitwise_and(mask, line_mask)

            if np.sum(intersection) > 0:
                img[np.where(intersection > 0)] = (0, 255, 255)  # yellow

    # ---- DISPLAY IMAGE ----
    cv2.imshow("Arrows + Line Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ---- SAVE IMAGE ----
    cv2.imwrite("output.jpg", img)
    print("Saved as output.jpg")


# RUN
detect_arrows_and_lines("input.jpg")
