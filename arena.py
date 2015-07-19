# -*- coding: utf-8 -*-
"""

Created on Sat Jul 18 08:38:23 2015

@author: akl
"""
import numpy as np
    
class arena:
    """Create arena object."""


    def __init__(self, arena_id):
        # Read walls image files
        self.arena_id = arena_id
        self.wall_images = np.load('walls/arena%d.npz' % arena_id)
        

    def wall1(self):
        """Get the first wall of the arena."""
        wall1 = self.wall_images['arr_0']
        return wall1


    def wall2(self):
        """Get the second wall of the arena."""
        wall2 = self.wall_images['arr_1']
        return wall2


    def wall3(self):
        """Get the third wall of the arena."""
        wall3 = self.wall_images['arr_2']
        return wall3


    def wall4(self):
        """Get the fourth wall of the arena."""
        wall4 = self.wall_images['arr_3']
        return wall4

    def maximum_length(self):
        """Get the maximum length of the arena. (x-direction)"""
        return len(self.wall1())

    def maximum_width(self):
        """Get the maximum width of the arena. (y-direction)"""
        return len(self.wall2())
