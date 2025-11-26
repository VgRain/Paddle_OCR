import fitz  # PyMuPDF

pdf_path = "drawing.pdf"
output_path = "weld_boxes_marked.pdf"

doc = fitz.open(pdf_path)

def is_horizontal(rect):
    width = rect.width
    height = rect.height
    return width > height * 1.8    # rectangle significantly wider than tall

for page_num, page in enumerate(doc):
    shapes = page.get_drawings()
    weld_boxes = []

    for shape in shapes:
        if "rect" in shape:
            rect = fitz.Rect(shape["rect"])

            # 1. Check if rectangle is horizontal
            if not is_horizontal(rect):
                continue

            # 2. Extract text inside this rectangle
            text_inside = page.get_textbox(rect).strip()

            # 3. Classify as weld or non-weld
            if "W" in text_inside.upper():   # weld box
                weld_boxes.append((rect, "WELD"))
            else:
                weld_boxes.append((rect, "RECT"))

    # Draw marks on the PDF
    for rect, kind in weld_boxes:
        if kind == "WELD":
            color = (1, 0, 0)     # red highlight for weld
        else:
            color = (0, 0, 1)     # blue for normal rectangle

        # highlight box
        page.add_rect_annot(rect)
        annot = page.add_rect_annot(rect)
        annot.set_colors(stroke=color)
        annot.update()

doc.save(output_path)
print("Done. Saved:", output_path)
