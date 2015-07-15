# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:13:41 2015

@author: akl

This script is for generating the wall images and storing them in files.

 _____________________
|        wall1        |
|                     |
|w                   w|
|a                   a|
|l                   l|
|l                   l|
|4                   2|
|                     |
|        wall3        |
 ---------------------
"""

import numpy as np
import random


def create_wall(width):
    """
    Initalize wall pixel values, depending on the width of the
    wall.
    """

    # Wall array represents black and white values per pixel
    wall = np.empty([width], dtype=int)
    colors = ['black', 'white']
    color = random.choice(colors)
    i = 0
    black = True

    # Create 1 row of pixel values
    while (width - i) > 50:
        # Stripe width lies between 5 and 50 mm
        stripeWidth = np.random.randint(5, 50)
        if color == 'black':
            fill_wall(wall, 0, i, stripeWidth)
            color = 'white'
        else:
            fill_wall(wall, 255, i, stripeWidth)
            color = 'black'
        i = i + stripeWidth

    if black:
        fill_wall(wall, 0, i, width - i)
    else:
        fill_wall(wall, 255, i, width - i)

    return wall


def fill_wall(wall, color, current_pos, width):
    """
    Modify values in the wall array based of the color and the extension
    value.
    """

    for i in range(current_pos, current_pos + width):
        wall[i] = color


if __name__ == '__main__':

    # Box dimensions 1000mm x 400mm
    wall1 = create_wall(800)
    wall2 = create_wall(432)
    wall3 = create_wall(800)
    wall4 = create_wall(432)

    np.savez('arena3', wall1, wall2, wall3, wall4)
