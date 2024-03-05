import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("./data/cv04c_robotC.bmp", cv2.IMREAD_GRAYSCALE) / 255

filters = ["./data/cv04c_filtDP.bmp", "./data/cv04c_filtDP1.bmp", "./data/cv04c_filtHP.bmp", "./data/cv04c_filtHP1.bmp"]
matrices = [cv2.imread(x, cv2.IMREAD_GRAYSCALE) / 255 for x in filters]
results = [cv2.filter2D(image, -1, x) for x in matrices]
spectrums = [np.log(np.abs(np.fft.fftshift(np.fft.fft2(x)))) for x in results]

rows = len(filters)
cols = 4

for k in range(rows):
    plt.subplot(rows, cols, (rows * k) + 1)
    plt.imshow(image, cmap="gray")

    plt.subplot(rows, cols, (rows * k) + 2)
    plt.imshow(matrices[k], cmap='gray')

    plt.subplot(rows, cols, (rows * k) + 3)
    plt.imshow(results[k], cmap='gray')

    plt.subplot(rows, cols, (rows * k) + 4)
    plt.imshow(spectrums[k], cmap='jet')

plt.show()
