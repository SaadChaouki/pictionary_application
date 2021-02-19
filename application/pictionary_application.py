from tkinter import *
from pyscreenshot import grab
import numpy as np
import matplotlib.pyplot as plt
import PIL.ImageOps

class Pictionary:
    def __init__(self, master):
        self.master = master
        self.color_fg = 'black'
        self.color_bg = 'white'
        self.old_x = None
        self.old_y = None
        self.widgets()
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset)

    def paint(self, e):
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, e.x, e.y, width=4, fill=self.color_fg, capstyle=ROUND,
                                    smooth=True)
        self.old_x = e.x
        self.old_y = e.y

    def reset(self, e):
        self.old_x = None
        self.old_y = None
        self.screenshot()
        print('Awaiting Input ...')

    def processImage(self, img, display=False):
        # Transform to array
        arrayPicture = np.array(img).sum(axis=2) + 1
        # Log picture to reduce effect of blur
        loggedPicture = np.log(arrayPicture)
        # Flatten
        flattenedPicture = loggedPicture.flatten()
        # Scale
        flattenedPicture /= max(flattenedPicture)
        # Set max to 255
        flattenedPicture *= 255
        # Round
        finalPicture = np.int0(flattenedPicture)
        # Displaying
        if display:
            plt.axis(False);
            plt.imshow(loggedPicture, cmap='Greys')
            plt.show()


        return finalPicture

    def screenshot(self):
        # Get Position
        x = self.master.winfo_rootx() + self.canvas.winfo_x()
        y = self.master.winfo_rooty() + self.canvas.winfo_y()

        # Updating the size
        x1 = x + self.canvas.winfo_width() - 10
        y1 = y + self.canvas.winfo_height() - 10

        # Screenshot
        screenshot = PIL.ImageOps.invert(grab(bbox=(x, y, x1, y1)).resize((28, 28)))

        # Process Picture
        processedImage = self.processImage(screenshot)

        print('Drawing Grabbed.')

        # Predict
        self.predict(processedImage)


    def predict(self, img):
        print('Predicting')
        print(list(img))
        modelPrediction = 'Bike'
        probability = .9
        predictionText = f'Model Prediction: {modelPrediction} - Probability : {probability}'
        # self.canvas.create_text(200, 10, font="Times 20 italic bold", text=predictionText, tag = "text")
        print('Updated Predictions')

    def clear(self):
        self.canvas.delete(ALL)

    def widgets(self):
        # Setting size
        self.canvas = Canvas(self.master, width=800, height=800, bg=self.color_bg, )
        self.canvas.pack(fill=BOTH, expand=True)

        menu = Menu(self.master)
        self.master.config(menu=menu)

        optionmenu = Menu(menu)
        menu.add_cascade(label='Options', menu=optionmenu)
        optionmenu.add_command(label='Clear', command=self.clear)