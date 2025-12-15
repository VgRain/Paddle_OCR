import cv2
import numpy as np
import math

class DimensionLineExtractor:
    def __init__(self, img):
        self.img = img
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, self.bw = cv2.threshold(self.gray, 200, 255, cv2.THRESH_BINARY_INV)

    # -----------------------------
    # 1. Get line direction from ROI
    # -----------------------------
    def _get_direction(self, roi):
        edges = cv2.Canny(roi, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180,
            threshold=30,
            minLineLength=20,
            maxLineGap=5
        )
        if lines is None:
            return None

        x1,y1,x2,y2 = lines[0][0]
        return np.array([x2-x1, y2-y1])

    # -----------------------------
    # 2. Walk along the line
    # -----------------------------
    def _walk_line(self, start_pt, direction, step=5, max_len=1500):
        x,y = start_pt
        dx,dy = direction / np.linalg.norm(direction)
        last = (x,y)

        for _ in range(max_len):
            nx = int(x + dx*step)
            ny = int(y + dy*step)

            if nx < 0 or ny < 0 or nx >= self.bw.shape[1] or ny >= self.bw.shape[0]:
                break

            if self.bw[ny, nx] == 0:
                break

            last = (nx, ny)
            x,y = nx,ny

        return last

    # -----------------------------
    # 3. Arrow detection (clean)
    # -----------------------------
    def _detect_arrow(self, x, y, size=35):
        roi = self.bw[
            max(0,y-size):min(self.bw.shape[0],y+size),
            max(0,x-size):min(self.bw.shape[1],x+size)
        ]

        # remove line pixels
        cv2.circle(roi, (size,size), 15, 0, -1)

        contours,_ = cv2.findContours(
            roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            if cv2.contourArea(cnt) < 80:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04*peri, True)
            if len(approx) == 3:
                return True
        return False

    # -----------------------------
    # PUBLIC API
    # -----------------------------
    def extract(self, bbox):
        x,y,w,h = bbox
        roi = self.bw[y:y+h, x:x+w]

        direction = self._get_direction(roi)
        if direction is None:
            return None

        # center of ROI
        cx = x + w//2
        cy = y + h//2

        # extend both directions
        end1 = self._walk_line((cx,cy), direction)
        end2 = self._walk_line((cx,cy), -direction)

        arrow1 = self._detect_arrow(*end1)
        arrow2 = self._detect_arrow(*end2)

        return {
            "line": (end1, end2),
            "arrow_start": arrow1,
            "arrow_end": arrow2
        }
