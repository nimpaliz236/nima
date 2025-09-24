import cv2
import numpy as np

# --- ۱. خواندن تصویر در حالت خاکستری ---
img = cv2.imread("nima.jpg", cv2.IMREAD_GRAYSCALE)

# --- ۲. ساخت کرنل (ماسک) ---
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

# --- ۳. عملیات مورفولوژیکی ---
erosion   = cv2.erode(img, kernel, iterations=1)                       # فرسایش
dilation  = cv2.dilate(img, kernel, iterations=1)                      # اتساع
opening   = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)              # باز کردن
closing   = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)             # بستن
gradient  = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)          # گرادیان
tophat    = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)            # کلاه سفید
blackhat  = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)          # کلاه سیاه
#
# Erosion (فرسایش) → مرزهای سفید رو نازک می‌کنه، نقاط ریز سفید حذف می‌شن.
#
# Dilation (اتساع) → مرزهای سفید رو کلفت می‌کنه، سوراخ‌های ریز پر می‌شن.
#
# Opening (بازکردن = erosion بعد dilation) → نویز سفید ریز حذف می‌شه ولی شکل اصلی می‌مونه.
#
# Closing (بستن = dilation بعد erosion) → سوراخ‌های مشکی کوچک داخل شکل پر می‌شن.
#
# Gradient (گرادیان) → تفاوت بین dilation و erosion رو نشون می‌ده → فقط مرزها سفید می‌شن.
#
# Top-hat (کلاه سفید) → بخش‌هایی از تصویر اصلی که بعد از opening حذف شدن رو نشون می‌ده → یعنی نویزهای روشن.
#
# Black-hat (کلاه سیاه) → بخش‌هایی از تصویر اصلی که بعد از closing اضافه شدن رو نشون می‌ده → یعنی سوراخ‌ها یا نویزهای تاریک

# --- ۴. نمایش نتایج ---
cv2.imshow("Original", img)
cv2.imshow("Erosion", erosion)
cv2.imshow("Dilation", dilation)
cv2.imshow("Opening", opening)
cv2.imshow("Closing", closing)
cv2.imshow("Gradient", gradient)
cv2.imshow("Top-hat", tophat)
cv2.imshow("Black-hat", blackhat)

cv2.waitKey(0)
cv2.destroyAllWindows()
