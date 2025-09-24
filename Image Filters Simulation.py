import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('nima.jpg')

blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
median_image = cv2.medianBlur(image, 5)
laplacian_image = cv2.Laplacian(image, cv2.CV_64F)


plt.subplot(1, 4, 1), plt.imshow(image), plt.title('Original Image')
plt.subplot(1, 4, 2), plt.imshow(blurred_image), plt.title('Gaussian Blur')
plt.subplot(1, 4, 3), plt.imshow(median_image), plt.title('Median Blur')
plt.subplot(1, 4, 4), plt.imshow(laplacian_image, cmap='gray'), plt.title('Laplacian Filter')
plt.show()
