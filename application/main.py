from application.pictionary_application import Pictionary
from tkinter import Tk

if __name__ == '__main__':
    root = Tk()
    Pictionary(root)
    root.title('Pictionary - Try to beat me!')
    root.mainloop()

