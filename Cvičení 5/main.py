from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

def crop(image, x, y, size):
    top = max(y - size // 2, 0)
    bottom = min(y + size // 2 + 1, image.shape[0])
    left = max(x - size // 2, 0)
    right = min(x + size // 2 + 1, image.shape[1])

    return image[top:bottom, left:right]

def average_with_rotating_mask(image):
    size = 3
    copy = np.zeros(image.shape)

    # loop over the image, pixel by pixel
    for y in range(0, image.shape[0]):
        for x in range(0, image.shape[1]):
            mask_1 = crop(image, x - 1, y - 1, size)
            mask_2 = crop(image, x - 1, y + 0, size)
            mask_3 = crop(image, x - 1, y + 1, size)
            mask_4 = crop(image, x,     y - 1, size)
            mask_5 = crop(image, x,     y + 1, size)
            mask_6 = crop(image, x + 1, y - 1, size)
            mask_7 = crop(image, x + 1, y + 0, size)
            mask_8 = crop(image, x + 1, y + 1, size)

            masks = [mask_1, mask_2, mask_3, mask_4, mask_5, mask_6, mask_7, mask_8]
            lowest_variance_index = np.argmin([np.var(x) for x in masks])

            copy[y, x] = np.mean(masks[lowest_variance_index])
    
    return copy


image = cv2.imread(Path("./data/cv05_robotS.bmp").as_posix(), cv2.IMREAD_GRAYSCALE)
kernel = 1/9 * np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
])

mean = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
rotation = average_with_rotating_mask(image)
median = cv2.medianBlur(image, 3)

rows = 2
cols = 4

plt.subplot(rows, cols, 1)
plt.imshow(image, cmap="gray")
plt.title("Vstup")

plt.subplot(rows, cols, 2)
plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(image)))), cmap='jet')
plt.title("Spektrum")

plt.subplot(rows, cols, 3)
plt.imshow(mean, cmap="gray")
plt.title("Výsledek - prosté průměrování")

plt.subplot(rows, cols, 4)
plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(mean)))), cmap='jet')
plt.title("Spektrum - prosté průměrování")

plt.subplot(rows, cols, 5)
plt.imshow(rotation, cmap="gray")
plt.title("Výsledek - rotující maska")

plt.subplot(rows, cols, 6)
plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(rotation)))), cmap='jet')
plt.title("Spektrum - rotující maska")

plt.subplot(rows, cols, 7)
plt.imshow(median, cmap="gray")
plt.title("Výsledek - medián")

plt.subplot(rows, cols, 8)
plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(median)))), cmap='jet')
plt.title("Spektrum - medián")


plt.show()