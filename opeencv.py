import cv2 as cv


# تعریف تابع rescaleframe
# def rescaleframe(frame, scale=0.75):
#     width = int(frame.shape[1] * scale)
#     height = int(frame.shape[0] * scale)
#     dimensions = (width, height)
#     return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


# خواندن و ریسایز تصویر
img = cv.imread("1.jpg")
cv.imshow("Original", img)
# if img is None:
#     print("Error: Cannot read image file.")
#     exit()
#
# resized_image = rescaleframe(img)
# cv.imshow("Original Image", img)
# cv.imshow("Resized Image", resized_image)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# # خواندن ویدئو
# capture = cv.VideoCapture('1.mp4')
# if not capture.isOpened():
#     print("Error: Cannot open video file.")
#     exit()
#
# while True:
#     isTrue, frame = capture.read()
#
#     if not isTrue:  # وقتی ویدئو تموم شد یا فریم خالی بود
#         print("End of video or cannot read frame.")
#         break
#
#     frame_resized = rescaleframe(frame)
#
#     cv.imshow('Video Frame', frame)
#     cv.imshow('Resized Video Frame', frame_resized)
#
#     if cv.waitKey(2) & 0xFF == ord('q'):
#         break
#
# capture.release()
# cv.destroyAllWindows()
#
#
ret,thtesh=cv.threshold(gray,127,255,cv.THRESH_BINARY)
blur=cv.GaussianBlur(img,(5,5),cv.BORDER_DEFAULT)
cv.imshow("blur",blur)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray)



