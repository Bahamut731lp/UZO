import cv2
import numpy as np

from tqdm import tqdm

image = cv2.imread("./data/cv04_rentgen.bmp")
equalized = image.copy()

histogram, _ = np.histogram(image[:,:,0].ravel(), bins=256, range=[0,256])

min_intensity = 0
lowest_intensity = 0
max_intensity = 255
width = image.shape[0]
height = image.shape[1]

for y in tqdm(range(0, width), desc="Histogram Equalization"):
    for x in range(0, height):
        intensity = image[y, x, 0]
        cumulative_intensity = np.sum(histogram[lowest_intensity:intensity+1])
        equalized[y, x] = (max_intensity / (width * height)) * cumulative_intensity

output = np.concatenate((image, equalized), axis=1)
cv2.imshow("Histogram EQ", output)
cv2.waitKey(0)