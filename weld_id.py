import cv2
import numpy as np

def detect_arrows(image_path, save_path="arrows_detected.jpg"):
    # Load & preprocess
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:  # ignore very small contours
            continue

        # Approximate the contour
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

        # Arrowheads usually have 5–8 sides
        if 5 <= len(approx) <= 8:

            # Bounding box
            x, y, w, h = cv2.boundingRect(cnt)

            # Aspect ratio check
            aspect_ratio = w / float(h)

            # Typical arrow head shape is slightly wide OR tall
            if 0.3 < aspect_ratio < 3.5:  
                # Draw bbox
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, "Arrow", (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # Save output
    cv2.imwrite(save_path, img)
    print("Saved:", save_path)

    # Display output
    cv2.imshow("Arrows", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example —
detect_arrows("input.jpg")
