import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('nima.jpg')
colors = ('b', 'g', 'r')  # Blue, Green, Red

plt.figure(figsize=(10, 5))

for i, color in enumerate(colors):
    hist = cv2.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(hist, color=color)

plt.title('Image Histogram')
plt.xlabel('Intensity Value')
plt.ylabel('Frequency')
plt.show()
