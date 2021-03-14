import numpy as np
import matplotlib.pyplot as plt
import random

def convertToImageArray(array, size = 28):
    '''
    Function to convert an image 784 array to a 2D array of size 28*28
    '''
    return np.reshape(array, (size, size))

def displayRandomImageGrid(data, columns = 6, rows = 5):
    '''
    Function to select a random subset of the data and draw the pictures in a grid.
    '''
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
    '''
    function to display a single image.
    '''
    counter = 0
    newArray = np.reshape(array, (size, size))
    plt.clf();
    plt.figure(num=None, figsize=(4, 4), dpi=80);
    plt.imshow(newArray, cmap='Greys', aspect='auto') ;
    plt.title(title)
    plt.axis(False);
    plt.show();