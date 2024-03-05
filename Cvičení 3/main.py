import cv2
import numpy as np
import math

def rotate(image: np.ndarray, degree: int):
    """
    Otočení obrázků kolem jeho středu o zadaný počet stupňů.
    """
    height, width = image.shape[:2]

    # Výpočet původního středu obrázku
    original_center = np.array([width // 2, height // 2])

    # Převedení úhlu na rozsah 0-360 stupňů
    degree = degree % 360
    rads = math.radians(degree)

    # Výpočet matice rotace
    rotation_matrix = np.array([
        [np.cos(rads), -np.sin(rads)],
        [np.sin(rads), np.cos(rads)]
    ])
    
    original_dimensions = np.array([[width, height]] * 2)
    
    # Výpočet nových rozměrů obrázku
    new_size = np.abs(rotation_matrix * original_dimensions).sum(axis=0)

    # Inverzní matice rotace
    inv_rotation_matrix = np.linalg.inv(rotation_matrix)

    # Výpočet nového středu obrázku
    center_x, center_y = new_size // 2

    # Vytvoření pole pro otáčený obrázek - defaultně černých pixelů
    image_rotation = np.zeros((int(new_size[1]), int(new_size[0]), 3), dtype=np.uint8)

    # Iterace přes každý pixel v otáčeném obrázku
    for i in range(image_rotation.shape[0]):
        for j in range(image_rotation.shape[1]):

            # Výpočet původních souřadnic pixelu v původním obrázku
            original_coords = np.dot(inv_rotation_matrix, np.array([i - center_y, j - center_x]))
            x, y = original_coords + original_center

            # Kontrola, zda jsou vypočítané souřadnice v mezích původního obrázku
            if 0 <= x < image.shape[0] and 0 <= y < image.shape[1]:
                image_rotation[i, j, :] = image[int(x), int(y), :]

    return image_rotation

if __name__ == "__main__":
    image = cv2.imread("./data/cv03_robot.bmp")

    for i in range(0,361,45):
        rotated_image = rotate(image, i)
        cv2.imshow("Otočený obrázek",rotated_image)
        cv2.waitKey(0)

