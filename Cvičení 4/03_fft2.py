import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("./data/cv04c_robotC.bmp", cv2.IMREAD_GRAYSCALE)
fft2 = np.fft.fft2(image)

spectrum = np.abs(fft2)
shifted_spectrum = np.fft.fftshift(spectrum)

# Vykreslení amplitudového spektra
plt.subplot(1, 2, 1)
plt.imshow(np.log(spectrum), cmap='jet')
plt.title('Amplitudové spektrum')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(np.log(shifted_spectrum), cmap='jet')
plt.title('Amplitudové spektrum s posunutými kvadranty')
plt.colorbar()

plt.show()