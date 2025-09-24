import numpy as np

f = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
window = np.ones(3) / 3  # [1/3, 1/3, 1/3]
moving_avg = np.convolve(f, window, mode='valid')
print(moving_avg)
