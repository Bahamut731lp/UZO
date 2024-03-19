from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(Path("./data/cv06_robotC.bmp").as_posix(), cv2.IMREAD_GRAYSCALE)

rows = 2
cols = 4

laplacian = cv2.filter2D(image, -1, np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
]))

# TODO: Tady někdo musí vymyslet, co Chaloupka myslí "modulem gradientu"
sobel_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
sobel_images = [cv2.filter2D(image, -1, np.rot90(sobel_kernel, x)) for x in range(8)]
sobel = np.max(sobel_images, axis=0)

kirsch_kernel = np.array([[3, 3, 3], [3, 0, 3], [-5, -5, -5]])
kirsch_images = [cv2.filter2D(image, -1, np.rot90(kirsch_kernel, x)) for x in range(8)]
kirsch = np.max(kirsch_images, axis=0)

plt.subplot(rows, cols, 1)
plt.imshow(image, cmap="gray")
plt.title("Vstup")

plt.subplot(rows, cols, 2)
plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(image)))), cmap='jet')
plt.title("Spektrum")

plt.subplot(rows, cols, 3)
plt.imshow(laplacian)
plt.colorbar()
plt.title("Výsledek - Laplace")

plt.subplot(rows, cols, 4)
plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(laplacian)))), cmap='jet')
plt.colorbar()
plt.title("Spektrum - Laplace")

plt.subplot(rows, cols, 5)
plt.imshow(sobel)
plt.colorbar()
plt.title("Výsledek - Sobel")

plt.subplot(rows, cols, 6)
plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(sobel)))), cmap='jet')
plt.colorbar()
plt.title("Spektrum - Sobel")

plt.subplot(rows, cols, 7)
plt.imshow(kirsch)
plt.colorbar()
plt.title("Výsledek - Kirsch")

plt.subplot(rows, cols, 8)
plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(kirsch)))), cmap='jet')
plt.colorbar()
plt.title("Spektrum - Kirsch")


plt.show()