import cv2
import numpy as np

# بارگذاری تصاویر به صورت خاکستری و ۸ بیتی
image = cv2.imread('nima.jpg', cv2.IMREAD_GRAYSCALE)
template = cv2.imread('nimap.jfif', cv2.IMREAD_GRAYSCALE)

# بررسی اینکه تصاویر به درستی بارگذاری شده‌اند
if image is None or template is None:
    print("Error loading images!")
    exit()

# اندازه تصویر و الگو را چاپ کن
print("Image shape:", image.shape)
print("Template shape:", template.shape)

# تطبیق الگو با تصویر
result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

# یافتن بهترین مکان تطابق
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# رسم مستطیل روی تصویر اصلی برای نشان دادن ناحیه تطابق
top_left = max_loc
bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

# نمایش تصویر با مستطیل تطابق
cv2.imshow('Matched Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
