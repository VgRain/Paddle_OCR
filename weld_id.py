import fitz               # PyMuPDF
import cv2
import numpy as np

class DimensionDetector:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.page = self.doc[0]
        self.text_blocks = []
        self.image = None

    # --------------------------------------------------------
    # 1. Extract all numbers (multi-orientation)
    # --------------------------------------------------------
    def extract_numbers(self):
        blocks = self.page.get_text("blocks")

        numbers = []
        for b in blocks:
            text = b[4]
            bbox = b[:4]

            # extract only pure numbers like: 20, 3, 10.5
            if self._is_number(text):
                numbers.append({"text": text, "bbox": bbox})

        self.text_blocks = numbers
        return numbers

    def _is_number(self, text):
        text = text.strip()
        return text.replace(".", "", 1).isdigit()

    # --------------------------------------------------------
    # 2. Render PDF page into image (for OpenCV)
    # --------------------------------------------------------
    def render_page(self, zoom=3):
        mat = fitz.Matrix(zoom, zoom)
        pix = self.page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img.reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        self.image = img
        return img

    # --------------------------------------------------------
    # 3. Detect horizontal & vertical lines
    # --------------------------------------------------------
    def detect_lines(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # HORIZONTAL lines
        horiz = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                                minLineLength=50, maxLineGap=5)

        # VERTICAL lines
        vert = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                               minLineLength=50, maxLineGap=5)

        return horiz, vert

    # --------------------------------------------------------
    # 4. For each number, check if it has a line near it
    # --------------------------------------------------------
    def find_nearby_line(self, number):
        x1, y1, x2, y2 = self._scale_bbox(number["bbox"])

        horiz, vert = self.detect_lines()

        number_y_center = (y1 + y2) / 2

        nearby_lines = []
        if horiz is not None:
            for l in horiz:
                xA, yA, xB, yB = l[0]
                # horizontal threshold
                if abs(yA - number_y_center) < 40:
                    nearby_lines.append(l[0])

        return nearby_lines

    def _scale_bbox(self, bbox, zoom=3):
        return [int(v * zoom) for v in bbox]

    # --------------------------------------------------------
    # 5. Detect arrowheads on a line (simple shape logic)
    # --------------------------------------------------------
    def has_two_arrows(self, line):
        x1, y1, x2, y2 = line

        roi = self._crop_line_region(x1, y1, x2, y2)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        arrow_count = 0
        for c in contours:
            area = cv2.contourArea(c)
            if 30 < area < 500:  # arrowhead area range
                arrow_count += 1

        return arrow_count >= 2

    def _crop_line_region(self, x1, y1, x2, y2):
        pad = 15
        x_min = max(0, min(x1, x2) - pad)
        x_max = min(self.image.shape[1], max(x1, x2) + pad)
        y_min = max(0, min(y1, y2) - pad)
        y_max = min(self.image.shape[0], max(y1, y2) + pad)
        return self.image[y_min:y_max, x_min:x_max]

    # --------------------------------------------------------
    # 6. Main pipeline
    # --------------------------------------------------------
    def run(self):
        self.extract_numbers()
        self.render_page()

        results = []

        for n in self.text_blocks:
            lines = self.find_nearby_line(n)
            for line in lines:
                if self.has_two_arrows(line):
                    results.append({
                        "number": n["text"],
                        "bbox": n["bbox"],
                        "line": line,
                        "has_arrows": True
                    })

        return results



det = DimensionDetector("drawing.pdf")

output = det.run()

for d in output:
    print("Number:", d["number"])
    print("Box:", d["bbox"])
    print("Line:", d["line"])
    print("Two Arrows:", d["has_arrows"])
    print("-------------")
