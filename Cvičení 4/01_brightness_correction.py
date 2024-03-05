import cv2
import numpy as np

images = {
    "./data/cv04_f01.bmp": "./data/cv04_e01.bmp",
    "./data/cv04_f02.bmp": "./data/cv04_e02.bmp"
}

for path_image, path_etalon in images.items():
    image = cv2.imread(path_image)
    etalon = cv2.imread(path_etalon)

    width = image.shape[0]
    height = image.shape[1]
    depth = image.shape[2]
    c = 255

    result = image.copy()

    for y in range(0, width):
        for x in range(0, height):
            for d in range(0, depth):
                result[y, x, d] = (c * image[y, x, d]) / (etalon[y, x, d])

    output = np.concatenate((image, etalon, result), axis=1)
    cv2.imshow(path_image, output)

cv2.waitKey(0)