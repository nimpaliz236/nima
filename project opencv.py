import cv2
import numpy as np
import matplotlib.pyplot as plt

# بارگذاری ویدیو
cap = cv2.VideoCapture('video.mp4')

# بررسی اینکه ویدیو به درستی باز شده
if not cap.isOpened():
    print("Error: Couldn't open video.")
    exit()

# خواندن اولین فریم
ret, frame1 = cap.read()
if not ret:
    print("Error: Couldn't read video frame.")
    exit()

# تبدیل اولین فریم به خاکستری
prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# ذخیره داده‌های حرکت
motion_data = []

while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    # تبدیل فریم به خاکستری
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # فیلتر Gaussian برای کاهش نویز
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # محاسبه تفاوت فریم‌ها
    diff = cv2.absdiff(prev_gray, gray)

    # آستانه‌گذاری برای شناسایی حرکت
    _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

    # محاسبه مساحت حرکت
    motion_area = np.sum(thresh > 0) / (thresh.shape[0] * thresh.shape[1])
    motion_data.append(motion_area)

    # نمایش فریم و نواحی حرکت
    cv2.imshow('Motion Detection', thresh)

    # بروزرسانی فریم قبلی
    prev_gray = gray

    # برای خروج از حلقه، دکمه 'q' را فشار بده
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# پایان کار و بستن ویدیو
cap.release()
cv2.destroyAllWindows()

# رسم داده‌های حرکت
plt.plot(motion_data)
plt.title('Motion Detection')
plt.xlabel('Frame')
plt.ylabel('Motion Area')
plt.show()
