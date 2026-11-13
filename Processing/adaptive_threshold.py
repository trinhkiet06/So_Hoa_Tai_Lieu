import cv2
import numpy as np


"""
    Thực hiện thuật toán phân ngưỡng thích ứng (Adaptive Thresholding) thủ công:
    Input Image -> Grayscale Conversion -> Local Mean/Gaussian Calculation 
    -> Threshold Matrix Evaluation -> Binarization -> Output Result

    Tham số:
        img            : ảnh đầu vào (ảnh màu BGR hoặc ảnh xám Grayscale từ OpenCV).
        max_value      : giá trị gán cho các pixel thỏa mãn điều kiện (thường là 255).
        method         : phương pháp tính ngưỡng cục bộ 
                         (cv2.ADAPTIVE_THRESH_MEAN_C hoặc cv2.ADAPTIVE_THRESH_GAUSSIAN_C).
        threshold_type : kiểu phân ngưỡng 
                         (cv2.THRESH_BINARY hoặc cv2.THRESH_BINARY_INV).
        block_size     : kích thước vùng lân cận để tính ngưỡng (bắt buộc phải là số nguyên lẻ > 1).
        C              : hằng số trừ đi từ giá trị trung bình cục bộ.

    Xử lý:
        grayscale conversion : chuyển đổi ảnh màu sang ảnh xám nếu ảnh đầu vào có 3 kênh màu.
        local calculation    : tính giá trị trung bình cục bộ bằng bộ lọc Blur hoặc GaussianBlur.
        threshold matrix     : tính ma trận ngưỡng T(x, y) bằng cách trừ đi hằng số C.
        binarization         : so sánh ma trận ảnh xám với ma trận ngưỡng để tạo ảnh nhị phân dst.

    Trả về:
        Ảnh nhị phân (dst) sau khi đã xử lý adaptive thresholding (kiểu uint8).
        Nếu block_size không hợp lệ hoặc định dạng ảnh sai, chương trình sẽ báo lỗi.
"""

def custom_adaptive_threshold(img, max_value, method, threshold_type, block_size, C):

    # Kiểm tra tính hợp lệ của block_size (phải là số lẻ và lớn hơn 1)
    if block_size < 3 or block_size % 2 == 0:
        raise ValueError("block_size phải là một số nguyên lẻ và lớn hơn hoặc bằng 3!")
    

    # Bước 1: Chuyển đổi ảnh đầu vào sang ảnh xám (Grayscale)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()


    # Bước 2: Tính toán giá trị trung bình cục bộ (Local Mean / Weighted Mean)
    if method == cv2.ADAPTIVE_THRESH_MEAN_C:
        # Sử dụng bộ lọc trung bình trượt (Mean Filter)
        local_mean = cv2.blur(gray, (block_size, block_size))


    elif method == cv2.ADAPTIVE_THRESH_GAUSSIAN_C:
        # Sử dụng bộ lọc trọng số phân phối chuẩn Gaussian (Gaussian Blur)
        local_mean = cv2.GaussianBlur(gray, (block_size, block_size), 0)
    else:
        raise ValueError("Method không hợp lệ! Chọn ADAPTIVE_THRESH_MEAN_C hoặc ADAPTIVE_THRESH_GAUSSIAN_C.")
    

    # Bước 3: Tính ma trận ngưỡng T(x, y) = local_mean - C
    # Ép kiểu sang float32 để tránh tràn số âm khi trừ đi hằng số C
    threshold_matrix = local_mean.astype(np.float32) - C


    # Bước 4: Phân ngưỡng nhị phân dựa trên loại threshold_type
    if threshold_type == cv2.THRESH_BINARY:
        # Nếu pixel lớn hơn ngưỡng T(x, y) thì gán max_value, ngược lại gán 0
        dst = np.where(gray > threshold_matrix, max_value, 0).astype(np.uint8)

        
    elif threshold_type == cv2.THRESH_BINARY_INV:
        # Ngược lại của THRESH_BINARY
        dst = np.where(gray > threshold_matrix, 0, max_value).astype(np.uint8)
        
        
    else:
        raise ValueError("threshold_type chỉ hỗ trợ cv2.THRESH_BINARY hoặc cv2.THRESH_BINARY_INV!")

    # Trả về kết quả ảnh nhị phân
    return dst