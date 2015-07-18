# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 08:38:23 2015

@author: akl
"""
import numpy as np

# Read walls image files
arena = 1
wall_images = np.load('walls/arena%d.npz' % arena)
wall1 = wall_images['arr_0']
wall2 = wall_images['arr_1']
wall3 = wall_images['arr_2']
wall4 = wall_images['arr_3']

wall_dict = {1: wall1, 2: wall2, 3: wall3, 4: wall4}

# Arena dimensions
x_max = len(wall1)
y_max = len(wall2)
