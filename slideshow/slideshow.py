''' tk_image_slideshow3.py
create a Tkinter image repeating slide show
tested with Python27/33  by  vegaseat  03dec2013
'''

import os
import sys
import Tkinter as tk
from itertools import cycle
from PIL import Image
from PIL import ImageTk

class App(tk.Tk):
    '''Tk window/label adjusts to size of image'''


    def __init__(self, image_files, x, y, delay):
        # the root will be self
        tk.Tk.__init__(self)
        # set x, y position only
        self.geometry('+{}+{}'.format(x, y))
        self.delay = delay
        # allows repeat cycling through the pictures
        # store as (img_object, img_name) tuple
        self.pictures = cycle((ImageTk.PhotoImage(Image.open(image)), image)
                              for image in image_files)
        self.picture_display = tk.Label(self)
        self.picture_display.pack()


    def show_slides(self):
        '''cycle through the images and show them'''
        # next works with Python26 or higher
        img_object, img_name = next(self.pictures)
        self.picture_display.config(image=img_object)
        # shows the image filename, but could be expanded
        # to show an associated description of the image
        self.title(img_name)
        self.after(self.delay, self.show_slides)


    def run(self):
        self.mainloop()


def main():

    plot_type = sys.argv[1]
    path = sys.argv[2]
    populations = []
    image_files = []

    content = os.listdir(path)
    content.sort()
    for item in content:
        if str(item).startswith('pop'):
            populations.append(item)

    for pop in populations:
        generations = os.listdir(path+'/'+pop)
        generations.sort()
        for g in range(len(generations)):
            image_files.append(path+'/'+pop+'/'+'gen%d' % (g+1)+'/'+plot_type+'.png')
    
    delay = 1000
    x = 100
    y = 50
    app = App(image_files, x, y, delay)
    app.show_slides()
    app.run()

if __name__ == '__main__':
    main()
