
import cv2
import numpy as np
from PIL import Image
"""
    Xuất hình ảnh tài liệu sau xử lý thành file PDF theo quy trình:
    Image Input -> Color Conversion -> A4 Page Creation
    -> Resize -> Center Alignment -> PDF Export

    Tham số:
        processed_images : danh sách các ảnh tài liệu sau khi xử lý
                           (ảnh Grayscale hoặc ảnh màu từ OpenCV).
        output_path      : đường dẫn và tên file PDF cần xuất.

    Xử lý:
        color conversion : chuyển đổi ảnh từ BGR sang RGB để tương thích với PIL.
        A4 page          : tạo trang giấy A4 với độ phân giải 300 DPI.
        resize           : thay đổi kích thước ảnh phù hợp với trang A4,
                           giữ nguyên tỷ lệ để tránh biến dạng.
        center alignment : căn giữa ảnh trên trang giấy, tạo khoảng trắng xung quanh.
        PDF export       : lưu một hoặc nhiều ảnh thành file PDF.

    Trả về:
        File PDF chứa hình ảnh tài liệu đã xử lý.
        Nếu không có ảnh đầu vào hoặc quá trình xuất gặp lỗi,
        chương trình sẽ thông báo lỗi.
"""

def export_to_pdf(processed_images, output_path):

    # Kich thuoc giay A4 tai 300 DPI
    A4_WIDTH = 2480
    A4_HEIGHT = 3508

    # Khoang cach le trang
    MARGIN = 150

    pdf_pages = []

    # Duyet qua tung anh
    for img in processed_images:

        # Buoc 1: Chuyen anh OpenCV sang PIL
        if len(img.shape) == 2:
            pil_img = Image.fromarray(img).convert("RGB")

        elif img.shape[2] == 3:
            rgb_img = cv2.cvtColor(
                img, cv2.COLOR_BGR2RGB
            )
            pil_img = Image.fromarray(rgb_img)

        elif img.shape[2] == 4:
            rgba_img = cv2.cvtColor(
                img, cv2.COLOR_BGRA2RGBA
            )
            pil_img = Image.fromarray(rgba_img).convert("RGB")

        else:
            raise ValueError("Dinh dang anh khong hop le!")

        # Buoc 2: Tao trang giay A4 mau trang
        page = Image.new(
            "RGB",
            (A4_WIDTH, A4_HEIGHT),
            "white"
        )

        # Buoc 3: Tinh kich thuoc toi da cua anh
        max_width = A4_WIDTH - 2 * MARGIN
        max_height = A4_HEIGHT - 2 * MARGIN

        # Thay doi kich thuoc nhung giu nguyen ti le
        pil_img.thumbnail(
            (max_width, max_height),
            Image.Resampling.LANCZOS
        )

        # Buoc 4: Can giua anh tren trang A4
        x = (A4_WIDTH - pil_img.width) // 2
        y = (A4_HEIGHT - pil_img.height) // 2

        page.paste(pil_img, (x, y))

        # Them trang vao danh sach PDF
        pdf_pages.append(page)

    # Buoc 5: Xuat file PDF
    if len(pdf_pages) == 0:
        raise ValueError("Khong co anh de xuat PDF!")

    pdf_pages[0].save(
        output_path,
        "PDF",
        resolution=300.0,
        save_all=True,
        append_images=pdf_pages[1:]
    )

    print("Xuat PDF thanh cong:", output_path)
