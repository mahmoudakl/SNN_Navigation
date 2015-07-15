# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 23:18:34 2015

@author: akl
"""

import numpy as np

if __name__ == '__main__':

    # Number of populations
    n = 1

    # n populations each comprising 60 individuals
    populations = np.random.randint(2, size=(n, 60, 10, 29))

    np.savez('population6', populations)
