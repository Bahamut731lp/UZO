"""
    Modul pro řešení úlohy 1 - viz zadání.pdf
"""
from pathlib import Path
from matplotlib import pyplot as plt
import cv2
import numpy as np

def preprocess(filepath):
    image_data = cv2.imread(filepath)
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    grayscale = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
    histogram = cv2.calcHist([grayscale], [0], None, [256], [0, 256])

    return {
        "histogram": histogram,
        "image": image_data
    }

def main():
    folder = Path("./data")
    images = folder.glob("*.jpg")
    data = [preprocess(x.as_posix()) for x in images]
    images = plt.figure("Porovnání (obrázky)")
    hists = plt.figure("Porovnání (histogramy)")
    dimension = len(data)
    
    for image_index, _ in enumerate(data):    
        image_distances = [
            cv2.compareHist(
                data[image_index]["histogram"],
                data[x]["histogram"],
                cv2.HISTCMP_INTERSECT 
            ) for x, _ in enumerate(data)
        ]

        np_image_distances = np.array(image_distances)
        indices = np.argsort(np_image_distances)[::-1]

        print(image_index, np_image_distances[indices])

        for index in range(len(indices)):
            plt.figure(images)
            plt.subplot(dimension, dimension, (image_index * (dimension)) + index + 1)
            plt.imshow(data[indices[index]]["image"])
            
            plt.figure(hists)
            plt.subplot(dimension, dimension, (image_index * (dimension)) + index + 1)
            plt.plot(data[indices[index]]["histogram"])

    images.show()
    hists.show()
    input("...")

if __name__ == "__main__":
    main()