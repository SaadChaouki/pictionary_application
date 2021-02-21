import numpy as np
import matplotlib.pyplot as plt
import random

def convertToImageArray(array, size = 28):
    counter = 0
    newArray = np.zeros((size, size))
    for i in range(0, size):
        for j in range(0, size):
            newArray[i, j] = array[counter]
            counter += 1
    return newArray

def displayRandomImageGrid(data, columns = 6, rows = 5):
    w = h = 10
    fig = plt.figure(figsize=(16, 10))
    for i in range(1, columns*rows +1):
        randomKey = random.choice(list(data.keys()))
        randomData = random.choice(data[randomKey])
        transformedData = convertToImageArray(randomData)
        img = np.random.randint(10, size=(h,w))
        ax = fig.add_subplot(rows, columns, i)
        ax.set_title(randomKey)
        ax.axis('off')
        plt.imshow(transformedData, cmap = 'Greys')
    plt.show()

def displayImage(array, size = 28, title = ''):
    counter = 0
    newArray = np.zeros((size, size))
    for i in range(0, size):
        for j in range(0, size):
            newArray[i, j] = array[counter]
            counter += 1
    plt.clf();
    plt.figure(num=None, figsize=(4, 4), dpi=80);
    plt.imshow(newArray, cmap='Greys', aspect='auto') ;
    plt.title(title)
    plt.axis(False);
    plt.show();