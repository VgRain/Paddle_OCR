import cv2
import numpy as np

class ArrowLineDetector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.edges = cv2.Canny(self.gray, 50, 150)

    # ----------------------------------------------------------
    # 1. Detect lines
    # ----------------------------------------------------------
    def detect_lines(self):
        lines = cv2.HoughLinesP(
            self.edges,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=40,
            maxLineGap=10
        )
        return [] if lines is None else lines[:, 0]

    # ----------------------------------------------------------
    # 2. Crop area around the line to inspect arrowheads
    # ----------------------------------------------------------
    def crop_line_region(self, x1, y1, x2, y2, pad=25):
        x_min = max(0, min(x1, x2) - pad)
        x_max = min(self.image.shape[1], max(x1, x2) + pad)
        y_min = max(0, min(y1, y2) - pad)
        y_max = min(self.image.shape[0], max(y1, y2) + pad)
        return self.image[y_min:y_max, x_min:x_max]

    # ----------------------------------------------------------
    # 3. Detect arrowheads in the cropped line region
    # ----------------------------------------------------------
    def has_two_arrows(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        arrow_count = 0
        for c in contours:
            area = cv2.contourArea(c)
            if 20 < area < 400:   # typical arrowhead size range
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.3 * peri, True)

                # Arrowheads look like triangles
                if len(approx) == 3:
                    arrow_count += 1

        return arrow_count >= 2

    # ----------------------------------------------------------
    # 4. Main logic: keep only lines with 2 arrowheads
    # ----------------------------------------------------------
    def get_lines_with_arrows(self):
        lines = self.detect_lines()
        valid_lines = []

        for (x1, y1, x2, y2) in lines:
            roi = self.crop_line_region(x1, y1, x2, y2)
            if self.has_two_arrows(roi):
                valid_lines.append((x1, y1, x2, y2))

        return valid_lines

    # ----------------------------------------------------------
    # 5. Display the good lines only
    # ----------------------------------------------------------
    def show_results(self, lines):
        out = self.image.copy()
        for (x1, y1, x2, y2) in lines:
            cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 3)

        cv2.imshow("Lines With Two Arrows", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ----------------------------------------------------------
# Usage
# ----------------------------------------------------------
if __name__ == "__main__":
    det = ArrowLineDetector("your_image.png")

    good_lines = det.get_lines_with_arrows()

    print("Detected lines with 2 arrows:")
    for l in good_lines:
        print(l)

    det.show_results(good_lines)
