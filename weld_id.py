vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,20))
vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,1))
horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

table_mask = cv2.add(horizontal_lines, vertical_lines)

