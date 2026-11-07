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
