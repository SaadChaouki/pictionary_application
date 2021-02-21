from tkinter import *
from pyscreenshot import grab
import numpy as np
import matplotlib.pyplot as plt
import PIL.ImageOps
import json
import random
from PIL import ImageTk, Image

from application.api_requests import RequestsAPI


class Pictionary():
    def __init__(self, master):
        self.master = master
        self.color_fg = 'black'
        self.color_bg = 'white'
        self.old_x = None
        self.old_y = None
        self.widgets()
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset)
        self.api = RequestsAPI()
        self.dictionary = json.load(open('resources/dictionary.json'))
        self.createPredictionWindow()
        self.predictionText = None

    def createPredictionWindow(self):
        self.window = Toplevel()
        self.window.title('Mimir Predictions')
        self.window.geometry("600x100")
        image1 = Image.open("resources/icon.png")
        image1 = image1.resize((75, 75), Image.ANTIALIAS)
        test = ImageTk.PhotoImage(image1)
        label1 = Label(self.window, image=test)
        label1.image = test
        label1.place(x=5, y=10)
        self.predictionText = Label(self.window, text=f"Welcome to Pictionary!!!", fg='black',
                                    font=("Open Sans", 20))
        self.predictionText.place(x=120, y=30)

    def updatePredictionWindow(self, text):
        if self.predictionText is not None:
            self.predictionText.destroy()
        self.predictionText = Label(self.window, text=text, fg='black', font=("Open Sans", 20))
        self.predictionText.place(x=120, y=30)

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
        csvImage = ','.join(img.astype(str))

        # Predict
        try:
            modelPrediction = self.dictionary[str(self.api.request_prediction(csvImage))]
            predictionText = f'I think this is {modelPrediction} ....           '
        except:
            predictionText = 'Oops, seems like Mimir is sleeping.           '

        self.updatePredictionWindow(predictionText)

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
