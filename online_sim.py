# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 14:54:34 2015

@author: akl

Module for Simulation Live demonstration.
"""

import matplotlib.pyplot as plt

import vision
import arena as ar

arena = ar.arena(1)

def get_axis_points(intersect, arena):
    """
    Get axis points to to draw view borders.

    @param intersect: view border tuple comprising visible wall number
                        and coordinate on that wall
    """

    if intersect[0] in (1, 3):
        x = intersect[1]
        y = arena.maximum_width() if intersect[0] == 1 else 0
    elif intersect[0] in (2, 4):
        y = intersect[1]
        x = arena.maximum_length() if intersect[0] == 2 else 0

    return x, y


def draw_borders(x_init, y_init, il, ir):
    """
    """

    x, y = get_axis_points(il)


def init(x_init, y_init, theta_init):
    """
    Initialization
    """

    ax1 = plt.subplot2grid((1, 8), (0, 0), colspan=6)
    ax2 = plt.subplot2grid((1, 8), (0, 6), colspan=2)

    plt.sca(ax1)
    plt.xlim(0, arena.maximum_length())
    plt.ylim(0, arena.maximum_width())
    plt.xticks([])
    plt.yticks([])
    plt.title("Arena")
    plt.plot(x_init, y_init, 'Dr')
    il, ir = vision.get_visible_wall_coordinates(x_init, y_init, theta_init,
                                                 arena)
    x, y = get_axis_points(il, arena)
    plt.plot([x_init, x], [y_init, y], '-b')
    x, y = get_axis_points(ir, arena)
    plt.plot([x_init, x], [y_init, y], '-b')
    r = vision.get_walls_view_ratio(il, ir, x_init, y_init, theta_init, arena)
    view = vision.get_view(x_init, y_init, il, ir, r, arena)
    view = vision.quadruple_pixels(view)
    view = vision.vertical_stack(view)

    plt.sca(ax2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(view, cmap=plt.get_cmap('gray'))

    #plt.show()