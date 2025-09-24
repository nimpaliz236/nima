import numpy as np
import cv2 as cv

blank=np.zeros((500,500,3),dtype='uint8')
cv.imshow('blank',blank)

blank[200:300, 300:400]=0,0,255
# cv.rectangle(blank,(200,200),(300,300),(0,251,55),thickness=1)
# cv.imshow('green',blank)
# cv.line(blank,(200,200),(300,300),(0,251,55),thickness=10)
# cv.imshow('yellow',blank)
cv.putText(blank,"hello",(10,20),cv.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)




cv.waitKey(0)