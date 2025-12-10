import cv2
import numpy as np

class ArrowDimensionDetector:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    # ----------------------------------------------------------
    # 1. Detect arrowheads (triangles)
    # ----------------------------------------------------------
    def detect_arrowheads(self):
        _, th = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        arrows = []

        for c in contours:
            area = cv2.contourArea(c)

            # filter noise and big shapes
            if area < 20 or area > 800:
                continue

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.25 * peri, True)

            # triangle → arrowhead
            if len(approx) == 3:
                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                arrows.append((cx, cy))

        return arrows

    # ----------------------------------------------------------
    # 2. Pair arrowheads and create a line between them
    # ----------------------------------------------------------
    def pair_arrows(self, arrows, max_dist=600):
        lines = []

        for i in range(len(arrows)):
            for j in range(i+1, len(arrows)):
                (x1, y1), (x2, y2) = arrows[i], arrows[j]

                # distance between arrowheads
                dist = np.hypot(x2 - x1, y2 - y1)

                # pair only close arrowheads (dimension line)
                if 30 < dist < max_dist:
                    lines.append((x1, y1, x2, y2))

        return lines

    # ----------------------------------------------------------
    # 3. Display all detected dimension lines
    # ----------------------------------------------------------
    def show_lines(self, lines):
        out = self.image.copy()
        for (x1, y1, x2, y2) in lines:
            cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 3)

        cv2.imshow("Dimension Lines", out)
        cv2.waitKey(0)

    # ----------------------------------------------------------
    # Main function
    # ----------------------------------------------------------
    def run(self):
        arrows = self.detect_arrowheads()

        if len(arrows) < 2:
            print("No arrow pair detected.")
            return []

        lines = self.pair_arrows(arrows)
        return lines
