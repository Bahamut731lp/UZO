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

def get_threshold(array):
    """Funkce pro výpočet prahu pro segmentaci obrázku z histogramu

    Args:
        array (_type_): Histogram

    Returns:
        Hodnota prahu
    """
    threshold = np.where((array[1:-1] < array[0:-2]) * (array[1:-1] < array[2:]))[0]
    threshold = threshold[0]

    return threshold

def get_centers_of_mass(image):
    """Funkce pro výpočet těžiště v segmentovaných oblastech

    Args:
        image: Matice s očíslovanými oblastmi. Pozadí je označeno číslem 0.

    Returns:
        Pole bodů těžišť.
    """
    points = []

    for i in range(2, np.max(image) + 1):
        copy = np.zeros_like(image)
        copy[image == i] = 1

        moments = cv2.moments(copy, True)
        points.append([int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])])

    return points

def get_region_values(image, points):
    values = []

    for point in points:
        region_number = image[point[1]][point[0]]
        number_of_pixels = len(np.argwhere(image == region_number))

        values.append({
            "point": point,
            "value": 5 if number_of_pixels > 4000 else 1
        })

    return values

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
                if all([
                    is_valid(new_x, new_y),     # Pokud je nový bod v rámci obrázku (abychom nepřetekli mimo něj)
                    not visited[new_x][new_y],  # Jestli jsme ten bod už náhodou před tím nebarvili
                    image[new_x][new_y] == 1    # Jestli bod není pozadí, aka dříve jsme tam vysegmentovali něco zajímavého
                ]):
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

_image = cv2.imread(Path("./data/cv07_segmentace.bmp").as_posix())
_image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)

red = np.float32(_image[:, :, 0])
green = np.float32(_image[:, :, 1])
blue = np.float32(_image[:, :, 2])

# Odečítám od 255, protože čím víc zelený je v obrázku, tím bělější ten pixel bude,
# ale mi potřebujeme přesnej opak - když je tam hodně zelený, má to bejt tmavý (páč je zelená v pozadí)
g = 255 - ((green * 255) / (red + green + blue))
g_hist = get_histogram(g, 10)
threshold = get_threshold(g_hist)

segmented_image = g.copy()
segmented_image[segmented_image < threshold] = 0
segmented_image[segmented_image >= threshold] = 1

regions = bfs_coloring(segmented_image)
points = get_centers_of_mass(regions)
coin_values = get_region_values(regions, points)

rows = 2
cols = 4

plt.subplot(rows, cols, 1)
plt.imshow(_image)
plt.title("Vstup")

plt.subplot(rows, cols, 2)
plt.imshow(g, cmap="gray")
plt.title("Zelená složka")

plt.subplot(rows, cols, 3)
plt.xlim(0, 255)
plt.ylim(0, 255)
plt.axis('square')
plt.plot(g_hist)
plt.vlines([threshold], 0, 255, "red")
plt.title("Histogram zelené složky")

plt.subplot(rows, cols, 4)
plt.title("Segmentace obrázku")
plt.imshow(segmented_image, cmap="gray")

plt.subplot(rows, cols, 5)
plt.title("Oblasti obrázku")
plt.imshow(regions, cmap='jet')

plt.subplot(rows, cols, 6)
plt.title("Těžiště oblastí")
plt.imshow(_image)
plt.scatter(*zip(*points), marker="+", color="red")

plt.subplot(rows, cols, 7)
plt.title("Hodnota oblastí")
plt.imshow(_image)
for coin in coin_values:
    print(coin)
    plt.text(coin.get("point")[0], coin.get("point")[1], coin.get("value"), color="red", fontsize=32, fontfamily="Consolas")

plt.show()