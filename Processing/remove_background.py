import cv2
import numpy as np


def remove_background(
    img_bgr,
    blur_size=5,
    canny_low=50,
    canny_high=150,
    close_size=7,
    min_area_ratio=0.1,
    crop=True,
):
    """
    Loại bỏ nền ảnh tài liệu bằng pipeline:
    Gray -> Blur -> Canny -> Closing -> Contour -> Convex Hull -> Mask -> Crop

    Tham số:
        blur_size       : kích thước kernel Gaussian (số lẻ), càng lớn càng bớt nhiễu nhưng biên bị mờ
        canny_low/high  : hai ngưỡng của Canny, càng thấp càng bắt nhiều biên
        close_size      : kích thước kernel Closing, càng lớn càng nối được các biên đứt xa
        min_area_ratio  : diện tích tối thiểu của tài liệu so với cả ảnh (0.1 = 10%)
        crop            : True thì cắt gọn ảnh theo khung tài liệu

    Trả về:
        (ảnh kết quả, mask). Nếu không tìm được tài liệu thì trả (ảnh gốc, None).
    """
    h, w = img_bgr.shape[:2]

    # ---- Bước 1: Chuyển sang ảnh xám ----
    # Canny chỉ làm việc trên 1 kênh độ sáng, nên không cần giữ 3 kênh màu.
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ---- Bước 2: Làm mờ Gaussian ----
    # Giảm nhiễu hạt và vân giấy/vân gỗ, tránh Canny bắt nhầm các biên nhỏ.
    gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

    # ---- Bước 3: Tìm biên bằng Canny ----
    # Ra ảnh nhị phân: pixel trắng là biên, pixel đen là vùng phẳng.
    edges = cv2.Canny(gray, canny_low, canny_high)

    # ---- Bước 4: Morphology Closing (giãn rồi co) ----
    # Nối các đoạn biên bị đứt quanh mép giấy để contour khép kín thành một vòng.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # ---- Bước 5: Tìm contour ngoài cùng ----
    # RETR_EXTERNAL chỉ lấy đường bao ngoài, bỏ qua các đường nằm bên trong
    # (chữ, bảng, hình trên tờ giấy).
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr, None  # không tìm được tài liệu

    # ---- Bước 6: Chọn contour có diện tích lớn nhất ----
    # Giả định tờ giấy là vật lớn nhất trong ảnh.
    c = max(contours, key=cv2.contourArea)

    # ---- Bước 7: Kiểm tra diện tích tối thiểu ----
    # Vùng quá nhỏ (dưới min_area_ratio) thì coi như không phải tài liệu, báo thất bại.
    if cv2.contourArea(c) < min_area_ratio * h * w:
        return img_bgr, None

    # ---- Bước 8: Convex Hull (bao lồi) ----
    # Làm đầy các chỗ lõm do mép giấy bị bóng đổ hoặc bị che, cho khung tài liệu trọn vẹn.
    hull = cv2.convexHull(c)

    # ---- Bước 9: Tạo mask ----
    # Mask nhị phân: 255 là vùng tài liệu, 0 là nền.
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [hull], -1, 255, thickness=cv2.FILLED)

    # ---- Bước 10: Tách nền ----
    # Tạo ảnh trắng toàn bộ, rồi chép lại pixel gốc ở vùng tài liệu.
    result = np.full_like(img_bgr, 255)
    result[mask == 255] = img_bgr[mask == 255]

    # ---- Bước 11: Cắt gọn theo khung chữ nhật bao quanh ----
    if crop:
        x, y, bw, bh = cv2.boundingRect(hull)
        result = result[y : y + bh, x : x + bw]

    return result, mask
