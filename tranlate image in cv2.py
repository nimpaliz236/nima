import numpy as np
import cv2

img=cv2.imread('1.jpg')

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edgs=cv2.Canny(gray,100,200,apertureSize=3)
#cv2.imshow('edgs',edgs)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# height, width = img.shape[:2]
# m=np.float32([[1,0,100],[0,1,50]])
# translated_hmg=cv2.warpAffine(img,m,(width,height))
# cv2.imshow('translated_hmg',translated_hmg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# def rotate(img,angle,rotpoint=None):
#     (height, width) = img.shape[:2]
#
#     if rotpoint is None:
#         rotpoint = (width//2, height//2)
#
#     rotmat = cv2.getRotationMatrix2D(rotpoint,angle,1)
#     dismentions=(width,height)
#
#     return cv2.warpAffine(img,rotmat,dismentions)
#
#
# rotate(img,45)
# cv2.imshow('rotated',rotate(img,45))
# cv2.waitKey(0)

contours , hierachies=cv2.findContours(edgs,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contour(s) found!')
img_copy = img.copy()
cv2.drawContours(img_copy, contours, -1, (0, 255, 0), 2)

# نمایش تصویر
cv2.imshow('Contours', img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()




