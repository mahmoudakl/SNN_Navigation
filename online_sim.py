# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 14:54:34 2015

@author: akl

Module for Simulation Live demonstration.
"""

import numpy as np
import matplotlib.pyplot as plt

import environment as env


def get_axis_points(intersect):
    """
    Get axis points to to draw view borders.

    @param intersect: view border tuple comprising visible wall number
                        and coordinate on that wall
    """

    if intersect[0] in (1, 3):
        x = intersect[1]
        y = env.y_max if intersect[0] == 1 else 0
    elif intersect[0] in (2, 4):
        y = intersect[1]
        x = env.x_max if intersect[0] == 2 else 0

    return x, y


def draw_borders(x_init, y_init, il, ir):
    """
    """

    x, y = get_axis_points(il)


def init(x_init, y_init, theta_init):
    """
    Initialization
    """

    ax1 = plt.subplot2grid((1, 7), (0, 0), colspan=6)
    #ax2 = plt.subplot2grid((1, 7), (0, 6), colspan=1)

    plt.sca(ax1)
    plt.xlim(0, env.x_max)
    plt.ylim(0, env.y_max)
    plt.xticks([])
    plt.yticks([])
    plt.title("Arena")
    plt.plot(x_init, y_init, 'Dr')

    il, ir = env.get_visible_wall_coordinates(x_init, y_init, theta_init)
    x, y = get_axis_points(il)
    plt.plot([x_init, x], [y_init, y], '-b')
    x, y = get_axis_points(ir)
    plt.plot([x_init, x], [y_init, y], '-b')
    r = env.get_walls_view_ratio(il, ir, x_init, y_init, theta_init)
    v = env.get_view(x_init, y_init, il, ir, r)

#    plt.sca(ax2)
#    plt.xticks([])
#    plt.yticks([])
#    v = env.view_image(v)
#    plt.imshow(v, cmap=plt.get_cmap('gray'))

    plt.show()