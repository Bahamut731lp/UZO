# https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html

from pathlib import Path
from collections import deque

import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_histogram(image, smoothing):
    """Funkce pro získání histogramu

    Args:
        image: Zdrojový obrázek
        smoothing: Síla vyhlazování výsledného histogramu

    Returns:
        Vyhlazený histogram
    """
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    normalized = histogram / np.max(histogram) * 255

    flattened = np.ndarray.flatten(normalized)
    smoothed = np.convolve(flattened, np.ones(smoothing)/smoothing, mode='same')

    return smoothed

def bfs_coloring(image):
    """Funkce pro barvení oblastí pomocí prohledávání do šířky

    Args:
        image: Binární obrázek k barvení

    Returns:
        Matice s očíslovanými oblastmi
    """
    rows, cols = image.shape
    visited = np.zeros_like(image).astype("bool")
    colors = np.zeros_like(image).astype("uint8")

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def is_valid(x, y):
        return 0 <= x < rows and 0 <= y < cols

    def bfs(x, y, color):
        queue = deque([(x, y)])
        visited[x][y] = True
        colors[x][y] = color

        # Zkoumáme všechny okolní body, dokud je v rámci obrázku, nebyli jsme na něm a je vysegmentovaný jako 1
        while queue:
            current_x, current_y = queue.popleft()

            for dx, dy in directions:
                new_x, new_y = current_x + dx, current_y + dy
                if is_valid(new_x, new_y) and not visited[new_x][new_y] and image[new_x][new_y] == 1:
                    queue.append((new_x, new_y))
                    visited[new_x][new_y] = True
                    colors[new_x][new_y] = color

    color_count = 0
    for i in range(rows):
        for j in range(cols):
            if not visited[i][j] and image[i][j] == 1:
                color_count += 1
                bfs(i, j, color_count)

    return colors

def get_centers_of_mass(image):
    """Funkce pro výpočet těžiště v segmentovaných oblastech

    Args:
        image: Matice s očíslovanými oblastmi. Pozadí je označeno číslem 0.

    Returns:
        Pole bodů těžišť.
    """
    points = []

    for i in range(1, np.max(image) + 1):
        copy = np.zeros_like(image)
        copy[image == i] = 1

        # Pokud není "celé zrno", tak nepočítáme
        if np.sum(copy) < 90:
            continue

        moments = cv2.moments(copy, True)
        points.append([int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])])

    return points

image = cv2.imread(Path("./data/cv10_mince.jpg").as_posix())
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blackhat = cv2.morphologyEx(grayscale, cv2.MORPH_BLACKHAT, np.ones((3, 3), np.uint8), iterations=20)
blackhat_hist = get_histogram(blackhat, 3)

blackhat = cv2.medianBlur(blackhat, ksize=7)
threshold = 7.5

segmentation = np.zeros_like(blackhat)
segmentation[blackhat < threshold] = 0
segmentation[blackhat >= threshold] = 1

rows = 3
cols = 4

morph = segmentation.copy()
morph = cv2.erode(morph, np.ones((3, 3), np.uint8))
morph = cv2.dilate(morph, np.ones((3, 3), np.uint8), iterations=5)

dist_transform = cv2.distanceTransform(morph, cv2.DIST_L2, 3)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

sure_fg = np.uint8(sure_fg)
morph = np.uint8(morph)
unknown = cv2.subtract(morph * 255, sure_fg)

ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown==255] = 0

outlined = image.copy()
markers_water = cv2.watershed(image, markers)
outlined[markers_water == -1] = [0, 255, 0]

points = get_centers_of_mass(markers)[1:]
print(points)

plt.subplot(rows, cols, 1)
plt.imshow(image, cmap="gray")
plt.title("Vstup")

plt.subplot(rows, cols, 2)
plt.imshow(blackhat, cmap="gray")
plt.title("Blackhat")

plt.subplot(rows, cols, 3)
plt.plot(blackhat_hist)
plt.vlines([threshold], 0, 255, "red")
plt.title(f"Blackhat Histogram - Práh {threshold}")

plt.subplot(rows, cols, 4)
plt.imshow(segmentation, cmap="gray")
plt.title("Segmentace")

plt.subplot(rows, cols, 5)
plt.imshow(morph, cmap="gray")
plt.title("Eroze + Dilatace")

plt.subplot(rows, cols, 6)
plt.imshow(sure_fg, cmap="gray")
plt.title("Sure FG")

plt.subplot(rows, cols, 7)
plt.imshow(markers, cmap="jet")
plt.title("Markers")

plt.subplot(rows, cols, 8)
plt.imshow(unknown, cmap="jet")
plt.title("Unknown")

plt.subplot(rows, cols, 9)
plt.imshow(outlined)
plt.scatter(*zip(*points), marker="+", color="green")
plt.title("Outlines")

plt.show()