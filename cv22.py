import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img=cv.imread("e.png")
#cv.imshow("Original", img)

# plt.imshow(img)
# plt.show()


blank = np.zeros(img.shape[:2],dtype="uint8")
#cv.imshow("Blank", blank)

# gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# #cv.imshow("gray",gray)

# blur=cv.GaussianBlur(img,(5,5),cv.BORDER_DEFAULT)
# #cv.imshow("blur",blur)

# ret,thresh=cv.threshold(gray,127,255,type=cv.THRESH_BINARY)
# cv.imshow("thresh",thresh)

# contours,hierarchies=cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
# print(f'{len(contours)} contours found')
# cv.drawContours(blank,contours,-1,(255,0,0),1)
# cv.imshow("Blank",blank)

# gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow("gray",gray)

# hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
# cv.imshow("hsv",hsv)

# lab=cv.cvtColor(img,cv.COLOR_BGR2LAB)
# cv.imshow("lab",lab)

# #تبدیل bgr به rgb
# rgb=cv.cvtColor(img,cv.COLOR_BGR2RGB)
# cv.imshow("rgb",rgb)
# plt.imshow(rgb)
# plt.show()

# #hsv to bgr
# hsv_bgr=cv.cvtColor(hsv,cv.COLOR_BGR2HSV)
# cv.imshow("hsv_bgr",hsv_bgr)

# b,g,r=cv.split(img)
# blue=cv.merge([b,blank,blank])
# green=cv.merge([blank,g,blank])
# red=cv.merge([blank,blank,r])

# cv.imshow("b",blue)
# cv.imshow("g",green)
# cv.imshow("r",red)

# print(img.shape)
# print(b.shape)
# print(g.shape)
# print(r.shape)

# merged=cv.merge([b,g,r])
# cv.imshow("merged",merged)
#
# avarege=cv.blur(img,(3,3))
# cv.imshow("blur",avarege)
#
# gauss=cv.GaussianBlur(img,(3,3),0)
# cv.imshow("guass",gauss)
#
# median=cv.medianBlur(img,3)
# cv.imshow("median",median)
#
# bilateral=cv.bilateralFilter(img,9,15,15)
# cv.imshow("bilateral",bilateral)

# blank=np.zeros((400,400),dtype=np.uint8)
# rectange=cv.rectangle(blank.copy(),(30,30),(370,370),255,-1)
# circle=cv.circle(blank.copy(),(200,200),200,255,-1)
# cv.imshow("rectange",rectange)
# cv.imshow("circle",circle)
#
# bitwise_and=cv.bitwise_and(rectange,circle)
# cv.imshow("bitwise",bitwise_and)
#
# bitwise_or=cv.bitwise_or(rectange,circle)
# cv.imshow("bitwise_or",bitwise_or)

# mask=cv.circle(blank,(int(img.shape[1]/2),int(img.shape[0]/2)), 100, 255,-1)
# cv.imshow("Mask", mask)

# masked=cv.bitwise_and(img,img,mask=mask)
# cv.imshow("Masked", masked)

# gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow("Gray", gray)
#
# gray_hist=cv.calcHist([gray],[0],None,[256],[0,256])
#
# plt.figure()
# plt.title("Gray Hist")
# plt.xlabel("Bins")
# plt.ylabel("# of Pixels")
# plt.plot(gray_hist)
# plt.xlim([0,256])
# plt.show()

#colors=('b','g','r')
# for i,color in enumerate (colors):
#     hist=cv.calcHist([img],[i],None,[256],[0,256])
#     plt.plot(hist,color=color)
#     plt.xlim([0,256])
# plt.show()

# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow("gray",gray)

# threshold, thresh = cv.threshold(gray,50,255,cv.THRESH_BINARY)
# cv.imshow("thresh",thresh)

# threshold, thresh_inv = cv.threshold(gray,50,255,cv.THRESH_BINARY_INV)
# cv.imshow("thresh_inv",thresh_inv)

# adaptive_tresh=cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,3)
# cv.imshow("adaptive_tresh",adaptive_tresh)

# lap = cv.Laplacian(gray, cv.CV_64F)
# lap = np.uint8(np.absolute(lap))
# cv.imshow("laplacian",lap)
#
# sobelx = cv.Sobel(gray,cv.CV_64F,1,0)
# sobely = cv.Sobel(gray,cv.CV_64F,0,1)
# cv.imshow("sobelx",sobelx)
# cv.imshow("sobely",sobely)
# sobel_combined = cv.magnitude(sobelx, sobely)
# cv.imshow("sobel_combined", sobel_combined)
# canny = cv.Canny(gray,150,175)
# cv.imshow("canny",canny)

#---------------------------------------------
#تشخیص چهره
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("gray",gray)

hear_cascade=cv.CascadeClassifier("face.xml")
faces_rect=hear_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)
print(f'number of face found: {len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
cv.imshow("face",img)

cv.waitKey(0)
cv.destroyAllWindows()