import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# --- خواندن تصویر در حالت خاکستری (یک کانال) ---
img = cv2.imread("nima.jpg", cv2.IMREAD_GRAYSCALE)


# --- تعریف کرنل‌ها (فیلترها) ---

# کرنل شارپ کردن (پیکسل وسط تقویت میشه و همسایه‌ها منفی میشن → تصویر واضح‌تر میشه)
kernel_sharpen = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

# کرنل لبه‌یابی (اختلاف شدت نور رو برجسته می‌کنه → لبه‌ها پیدا میشن)
kernel_edge = np.array([[-1, -1, -1],
                        [-1,  8, -1],
                        [-1, -1, -1]])

# کرنل بلور (میانگین‌گیری از 25 پیکسل اطراف → تصویر تار میشه)
kernel_blur = np.ones((5,5), np.float32) / 25


# --- تابع کانولوشن دستی (خودمون با numpy پیاده‌سازی می‌کنیم) ---
def convolve2d(image, kernel):
    # برگردوندن کرنل 180 درجه (برای تعریف ریاضی کانولوشن)
    kernel = np.flipud(np.fliplr(kernel))

    # خروجی هم‌سایز با تصویر اصلی ولی نوع داده float (برای محاسبات)
    output = np.zeros_like(image, dtype=np.float32)

    # محاسبه پدینگ (برای اینکه گوشه‌های تصویر رو هم بشه حساب کرد)
    pad_y, pad_x = kernel.shape[0]//2, kernel.shape[1]//2
    padded = np.pad(image, ((pad_y,pad_y),(pad_x,pad_x)), mode='constant')

    # اسکن کل تصویر (پیکسل به پیکسل)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # گرفتن بخشی از تصویر به اندازه کرنل
            region = padded[y:y+kernel.shape[0], x:x+kernel.shape[1]]
            # ضرب نقطه‌ای و جمع مقادیر
            output[y, x] = np.sum(region * kernel)

    # محدود کردن مقادیر به بازه [0, 255] (فرمت تصویر)
    output = np.clip(output, 0, 255)

    # تبدیل به uint8 تا خروجی بشه یک تصویر استاندارد
    return output.astype(np.uint8)


# --- اعمال فیلترها با روش دستی (کانولوشن خودمون) ---
manual_sharpen = convolve2d(img, kernel_sharpen)   # شارپ
manual_edge = convolve2d(img, kernel_edge)         # لبه‌یابی
manual_blur = convolve2d(img, kernel_blur)         # بلور


# --- اعمال فیلترها با تابع آماده OpenCV (filter2D) ---
cv_sharpen = cv2.filter2D(img, -1, kernel_sharpen)
cv_edge = cv2.filter2D(img, -1, kernel_edge)
cv_blur = cv2.filter2D(img, -1, kernel_blur)

#مقایسه تایمی بین دو روش برای فیلتر شارپ
start = time.time()
manual_sh = convolve2d(img, kernel_sharpen)
print("Manual time sharp:", time.time() - start)

start = time.time()
cv2_result_sh = cv2.filter2D(img, -1, kernel_sharpen)
print("OpenCV time sharp:", time.time() - start)

#مقایسه تایمی بین دو روش برای فیلتر ادج
start = time.time()
manual_e = convolve2d(img, kernel_edge)
print("Manual time edge:", time.time() - start)

start = time.time()
cv2_result_edge = cv2.filter2D(img, -1, kernel_edge)
print("OpenCV time edge:", time.time() - start)

#مقایسه تایمی بین دو روش برای فیلتر بلور
start = time.time()
manual_b = convolve2d(img, kernel_blur)
print("Manual time blur:", time.time() - start)

start = time.time()
cv2_result_blur = cv2.filter2D(img, -1, kernel_sharpen)
print("OpenCV time blur:", time.time() - start)


# --- نمایش تصاویر برای مقایسه ---
titles = ["Original",
          "Manual Sharpen", "CV2 Sharpen",
          "Manual Edge", "CV2 Edge",
          "Manual Blur", "CV2 Blur"]

images = [img,
          manual_sharpen, cv_sharpen,
          manual_edge, cv_edge,
          manual_blur, cv_blur]

plt.figure(figsize=(20,10))
for i in range(len(images)):
    plt.subplot(3,3,i+1)             # 3×3 شبکه
    plt.imshow(images[i], cmap="gray")  # نمایش تصویر خاکستری
    plt.title(titles[i])             # عنوان هر عکس
    plt.axis("off")                  # مخفی کردن محورهای x و y

plt.show()

