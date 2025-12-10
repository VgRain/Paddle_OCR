import cv2
import numpy as np
import pytesseract
from pytesseract import Output

class DimensionDetector:

    def __init__(self):
        self.kernel = np.ones((3, 3), np.uint8)

    # ---------------------------------------------------------
    # 1. Extract numbers (0–9) with angle correction
    # ---------------------------------------------------------
    def extract_numbers(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # tess config for picking numbers only
        config = "--psm 6 -c tessedit_char_whitelist=0123456789()"
        data = pytesseract.image_to_data(gray, output_type=Output.DICT, config=config)

        numbers = []
        for i in range(len(data["text"])):
            txt = data["text"][i].strip()
            if txt == "":
                continue

            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]

            numbers.append({"text": txt, "bbox": (x, y, w, h)})

        return numbers

    # ---------------------------------------------------------
    # 2. Check if a horizontal line exists below the number
    # ---------------------------------------------------------
    def detect_line_below(self, img, bbox):
        x, y, w, h = bbox
        roi_y1 = y + h + 5
        roi_y2 = y + h + 40  # scan 40px below text

        roi = img[roi_y1:roi_y2, x-20:x+w+20]
        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Hough line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40, minLineLength=40, maxLineGap=5)

        return lines

    # ---------------------------------------------------------
    # 3. Detect arrowheads at both ends of a line
    # ---------------------------------------------------------
    def detect_arrows(self, line_img):
        gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
        th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        arrow_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20 or area > 3000:
                continue

            approx = cv2.approxPolyDP(cnt, 0.2 * cv2.arcLength(cnt, True), True)

            # arrowhead tends to be triangular (3 corners)
            if len(approx) == 3:
                arrow_count += 1

        return arrow_count >= 2

    # ---------------------------------------------------------
    # Master function
    # ---------------------------------------------------------
    def process(self, img):
        results = []

        numbers = self.extract_numbers(img)

        for item in numbers:
            bbox = item["bbox"]
            num = item["text"]

            lines = self.detect_line_below(img, bbox)

            if lines is not None:
                # Extract region around the detected line to check arrowheads
                x, y, w, h = bbox
                roi = img[y+h+5:y+h+40, x-40:x+w+40]

                has_arrows = self.detect_arrows(roi)

                results.append({
                    "number": num,
                    "bbox": bbox,
                    "line_found": True,
                    "arrows": has_arrows
                })
            else:
                results.append({
                    "number": num,
                    "bbox": bbox,
                    "line_found": False,
                    "arrows": False
                })

        return results


det = DimensionDetector()

img = cv2.imread("drawing.png")

output = det.process(img)

for item in output:
    print(item)
