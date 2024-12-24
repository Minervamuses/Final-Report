import cv2
import easyocr
import numpy as np

# 設定影像路徑與 Haar Cascade 路徑
image_path = "/mnt/c/Final/images/Cars54.png" # 我的環境為WSL(Linux)，替換成你執行環境下的路徑
cascade_path = "/mnt/c/Final/haarcascade_russian_plate_number.xml"  # 同上

# 讀取影像
img = cv2.imread(image_path)
if img is None:
    print("影像讀取失敗。")
    exit()

# 灰階處理
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 加載 Haar Cascade
car_plate_haar_cascade = cv2.CascadeClassifier(cascade_path)

# 偵測車牌
car_plate_rects = car_plate_haar_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

if len(car_plate_rects) > 0:
    for x, y, w, h in car_plate_rects:
        img2 = cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        car_plate_img = gray[y:y + h, x:x + w]

    # 顯示檢測到的車牌和裁切影像
    cv2.imshow("Detected Plate", img2)
    cv2.imshow("Cropped Plate", car_plate_img)
    cv2.waitKey(0)  
    #cv2.destroyAllWindows()
else:
    print("未檢測到車牌。")
    exit()

# 放大車牌影像
scale_percent = 150  # 放大比例
width = int(car_plate_img.shape[1] * scale_percent / 100)
height = int(car_plate_img.shape[0] * scale_percent / 100)
dim = (width, height)
resized_image = cv2.resize(car_plate_img, dim, interpolation=cv2.INTER_AREA)

# EasyOCR 
reader = easyocr.Reader(["en"])  
result = reader.readtext(resized_image)

# 提取車牌號碼
if result:
    text = " ".join([item[1] for item in result]) 
    text = text.replace(" ", "") 
    print()
    print("車牌號碼:", text)
    cv2.waitKey(0)  
else:
    print("未能識別車牌號碼")
