import fitz
import math
import re


def is_small_circle(path):
    """Check whether a path is a tiny circle used as a degree symbol."""
    if "c" not in path.get("items", {}):  
        return False

    # circle approx via 4 cubic bezier curves (most CAD PDFs)
    rect = path["rect"]
    w = rect[2] - rect[0]
    h = rect[3] - rect[1]

    # tiny circle: ~1–5 px typically
    return 0.5 < w < 8 and 0.5 < h < 8


pdf = "drawing.pdf"
doc = fitz.open(pdf)

results = []

for pno, page in enumerate(doc):
    # 1) extract all paths
    paths = page.get_drawings()

    # find all small circles (degree symbols)
    degree_symbols = []
    for p in paths:
        if is_small_circle(p):
            degree_symbols.append(p["rect"])

    # 2) extract text (chars)
    chars = page.get_text("chars")

    # detect numbers near circles
    for circle in degree_symbols:
        cx = (circle[0] + circle[2]) / 2
        cy = (circle[1] + circle[3]) / 2

        # find nearby numbers (within 40px)
        nearby_chars = [
            c for c in chars
            if abs(c[1] - cy) < 20 and abs(c[0] - cx) < 40 and c[4].isdigit()
        ]

        if not nearby_chars:
            continue

        # group them to form number (e.g., '4','5' => "45")
        x_sorted = sorted(nearby_chars, key=lambda c: c[0])
        number = "".join(c[4] for c in x_sorted)

        # return bounding boxes
        num_bbox = (
            min(c[0] for c in x_sorted),
            min(c[1] for c in x_sorted),
            max(c[2] for c in x_sorted),
            max(c[3] for c in x_sorted),
        )

        results.append({
            "text": number + "°",
            "page": pno + 1,
            "circle_bbox": circle,
            "number_bbox": num_bbox
        })

print(results)    for l in lines:
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
