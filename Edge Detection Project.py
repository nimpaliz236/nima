import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('nima.jpg', cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(image, 100, 200)

plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(1, 2, 2), plt.imshow(edges, cmap='gray'), plt.title('Edge Detection')
plt.show()
