# -*- coding: utf-8 -*-
"""

Created on Sat Jul 18 08:38:23 2015

@author: akl
"""
import numpy as np
    
class arena:
    """Create arena object."""


    def __init__(self, arena_id):
        """Load wall images of the corresponding arena."""

        # Read walls image files
        self.arena_id = arena_id
        self.wall_images = np.load('walls/arena%d.npz' % arena_id)
        self.wall1 = self.wall_images['arr_0']
        self.wall2 = self.wall_images['arr_1']
        self.wall3 = self.wall_images['arr_2']
        self.wall4 = self.wall_images['arr_3']
        self.maximum_length = len(self.wall1)
        self.maximum_width = len(self.wall2)