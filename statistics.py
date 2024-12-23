import cv2
import easyocr
import os

# 文件路徑
image_folder = "/mnt/c/Final/images"  # 替換為你的影像資料夾路徑
cascade_path = "/mnt/c/Final/haarcascade_russian_plate_number.xml"  # 替換為 Haar Cascade 文件路徑

# 加載 Haar Cascade
car_plate_haar_cascade = cv2.CascadeClassifier(cascade_path)

# 初始化統計變數
total_images = 0
success_count = 0
failure_detected_plate_count = 0
failure_OCR_count = 0

# 初始化 OCR
reader = easyocr.Reader(["en"], gpu=True)

# 記錄第一張成功和失敗影像
first_success_image = None
first_failure_detected_plate_image = None
first_failure_OCR_image = None

# 遍歷影像資料夾
for filename in os.listdir(image_folder):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        total_images += 1
        image_path = os.path.join(image_folder, filename)
        img = cv2.imread(image_path)
        if img is None:
            print(f"無法讀取影像: {filename}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 車牌檢測
        car_plate_rects = car_plate_haar_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        if len(car_plate_rects) == 0:
            failure_detected_plate_count += 1
            if first_failure_detected_plate_image is None:
                first_failure_detected_plate_image = gray.copy()  # 保存分割狀態的影像
            continue

        # 裁切車牌影像
        x, y, w, h = car_plate_rects[0]
        car_plate_img = gray[y:y + h, x:x + w]

        # OCR 辨識
        result = reader.readtext(car_plate_img)
        if result:
            text = " ".join([item[1] for item in result]).replace(" ", "")
            if text:  # 如果辨識到文字
                success_count += 1
                if first_success_image is None:
                    first_success_image = car_plate_img.copy()  # 保存分割狀態的影像
            else:
                failure_OCR_count += 1
                if first_failure_OCR_image is None:
                    first_failure_OCR_image = car_plate_img.copy()  # 保存分割狀態的影像
        else:
            failure_OCR_count += 1
            if first_failure_OCR_image is None:
                first_failure_OCR_image = car_plate_img.copy()  # 保存分割狀態的影像

# 輸出統計結果
total_failure_count = failure_detected_plate_count + failure_OCR_count
print(f"總圖片數量: {total_images}")
print(f"成功圖片數: {success_count}")
print(f"檢測失敗圖片數 (failure_detected_plate): {failure_detected_plate_count}")
print(f"OCR 辨識失敗圖片數 (failure_OCR): {failure_OCR_count}")
print(f"總失敗圖片數: {total_failure_count}")

# 顯示第一張成功影像
if first_success_image is not None:
    #print("第一張成功的影像：")
    cv2.imshow("Success detected example (Processed)", first_success_image)
    cv2.waitKey(0)

# 顯示第一張檢測失敗影像
if first_failure_detected_plate_image is not None:
    #print("第一張檢測失敗的影像：")
    cv2.imshow("Failure detected plate example (Processed)", first_failure_detected_plate_image)
    cv2.waitKey(0)

# 顯示第一張 OCR 失敗影像
if first_failure_OCR_image is not None:
    #print("第一張 OCR 辨識失敗的影像：")
    cv2.imshow("Failure OCR example (Processed)", first_failure_OCR_image)
    cv2.waitKey(0)

# 關閉所有顯示窗口
cv2.destroyAllWindows()
