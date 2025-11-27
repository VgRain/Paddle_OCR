import cv2
import numpy as np
import math

def extend_line(x1, y1, x2, y2, scale=2000):
    vx = x2 - x1
    vy = y2 - y1
    return (x1 - vx*scale, y1 - vy*scale, x2 + vx*scale, y2 + vy*scale)

def intersection(l1, l2):
    x1,y1,x2,y2 = l1
    x3,y3,x4,y4 = l2

    denom = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    if abs(denom) < 1e-6:
        return None

    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom

    return (int(px), int(py))

def detect_boxes_parallel_perpendicular(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                            minLineLength=50, maxLineGap=10)

    horizontal = []
    vertical = []

    # Split lines by orientation
    for l in lines:
        x1,y1,x2,y2 = l[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

        if abs(angle) < 10 or abs(angle) > 170:
            horizontal.append((x1,y1,x2,y2))
        elif abs(angle - 90) < 10 or abs(angle + 90) < 10:
            vertical.append((x1,y1,x2,y2))

    boxes = []

    # For every pair of horizontal and vertical lines -> box candidate
    for h1 in horizontal:
        for h2 in horizontal:
            if h1 is h2:
                continue

            for v1 in vertical:
                for v2 in vertical:
                    if v1 is v2:
                        continue

                    # Extend the lines
                    h1e = extend_line(*h1)
                    h2e = extend_line(*h2)
                    v1e = extend_line(*v1)
                    v2e = extend_line(*v2)

                    # Compute corners
                    p1 = intersection(h1e, v1e)
                    p2 = intersection(h1e, v2e)
                    p3 = intersection(h2e, v2e)
                    p4 = intersection(h2e, v1e)

                    if None in (p1,p2,p3,p4):
                        continue

                    box = np.array([p1,p2,p3,p4])

                    # Filter small or weird shapes
                    area = cv2.contourArea(box)
                    if area < 500:
                        continue

                    boxes.append(box)
                    cv2.polylines(img, [box], True, (0,255,0), 2)

    return img


# Run
input_path = "drawing.png"
output_path = "boxes_out.png"

res = detect_boxes_parallel_perpendicular(input_path)
cv2.imwrite(output_path, res)

print("Done:", output_path)
