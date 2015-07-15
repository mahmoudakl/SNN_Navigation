# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 17:43:28 2015

@author: akl

Building a simulation environment for the experiment described in the
paper by Floreano, D. and Mattiussi, C., Evolution od Spiking Neural
Controllers for Autonomous Vision-based Robots.

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
import matplotlib.pyplot as plt


# Read walls image files
arena = 1
wall_images = np.load('walls/arena%d.npz' % arena)
wall1 = wall_images['arr_0']
wall2 = wall_images['arr_1']
wall3 = wall_images['arr_2']
wall4 = wall_images['arr_3']

wall_dict = {1: wall1, 2: wall2, 3: wall3, 4: wall4}

# Box dimensions
x_max = len(wall1)
y_max = len(wall2)


def get_visible_wall_coordinates(x, y, theta):
    """
    Calculate the visible part of the walls to the robot based on its
    position and orientation and the visual angle of 36 degrees.

    @param x: Robot x position
    @param y: Robot y position
    @param theta: Robot orientation angle
    """

    # The two angles reiesenting the vision range
    theta = theta % (2*np.pi)
    theta_1 = theta + np.deg2rad(18)
    theta_2 = theta - np.deg2rad(18)
    vertical_dist = y_max - y
    horizontal_dist = x_max - x

    # Robot's orientation angle lies in the first quadrant
    if theta >= 0 and theta < np.pi/2:

        if theta_1 == np.pi/2:
            # Left limit is exactly 90 deg., visible is wall1
            intersect_left = 1, x
        elif theta_1 < np.pi/2:
            # Left limit less than 90 deg., visible may be wall1 or
            # wall2
            angle = np.pi/2 - theta_1
            opp = vertical_dist*np.sin(angle)/np.cos(angle)
            if opp + x <= x_max:
                # Left limit is on wall1
                intersect_left = 1, opp + x
            else:
                # Left limit is on wall 2
                opp = opp + x - x_max
                adj = opp*np.cos(angle)/np.sin(angle)
                intersect_left = 2, y_max - adj
        else:
            # Left limit exceeds 90 deg., visible may be wall1 or wall4
            angle = theta_1 - np.pi/2
            opp = vertical_dist*np.sin(angle)/np.cos(angle)
            if x - opp >= 0:
                # Left limit is on wall1
                intersect_left = 1, x - opp
            else:
                # Left limit is on wall4
                opp = np.abs(x - opp)
                adj = opp*np.cos(angle)/np.sin(angle)
                intersect_left = 4, y_max - adj

        if theta_2 == 0:
            # Right limit is exactly 0 deg., visible is wall2
            intersect_right = 2, y
        elif theta_2 > 0:
            # Right limit exceeds 0 deg., visible may be wall1 or
            # wall2
            adj = vertical_dist*np.cos(theta_2)/np.sin(theta_2)
            if adj + x <= x_max:
                # Right limit is on wall1
                intersect_right = 1, adj + x
            else:
                # Right limit on wall2
                angle = np.pi/2 - theta_2
                opp = adj + x - x_max
                adj = opp*np.cos(angle)/np.sin(angle)
                intersect_right = 2, y_max - adj
        else:
            # Right limit less than 0 deg., visible may be wall2 or
            # wall3
            angle = np.abs(theta_2)
            opp = horizontal_dist*np.sin(angle)/np.cos(angle)
            if y - opp >= 0:
                # Right limit on wall2
                intersect_right = 2, y - opp
            else:
                # Right limit on wall3
                opp = opp - y
                adj = opp*np.cos(angle)/np.sin(angle)
                intersect_right = 3, x_max - adj

    # Robot's orientation angle lies in the second quadrant
    elif theta >= np.pi/2 and theta < np.pi:

        if theta_1 == np.pi:
            # Left limit is exactly 180 deg., visible is wall4
            intersect_left = 4, y
        elif theta_1 < np.pi:
            # Left limit less than 180 deg., visible may be wall1 or
            # wall4
            angle = np.pi - theta_1
            adj = vertical_dist*np.cos(angle)/np.sin(angle)
            if x - adj >= 0:
                # Left limit is on wall1
                intersect_left = 1, x - adj
            else:
                # Left limit is on wall2
                adj = np.abs(x - adj)
                opp = adj*np.sin(angle)/np.cos(angle)
                intersect_left = 4, y_max - opp
        else:
            # Left limit excceeds 180 deg., visible may be wall3 or
            # wall4
            angle = theta_1 - np.pi
            opp = x*np.sin(angle)/np.cos(angle)
            if y - opp >= 0:
                # Left limit on wall4
                intersect_left = 4, y - opp
            else:
                # Left limit on wall3
                opp = opp - y
                adj = opp*np.cos(angle)/np.sin(angle)
                intersect_left = 3, adj

        if theta_2 == np.pi/2:
            # Right limit is exact;y 90 deg., visible is wall1
            intersect_right = 1, x
        elif theta_2 > np.pi/2:
            # Right limit more than 90 deg., visible my be wall1 or
            # wall4
            angle = theta_2 - np.pi/2
            opp = vertical_dist*np.sin(angle)/np.cos(angle)
            if x - opp >= 0:
                # Right limit is on wall1
                intersect_right = 1, x - opp
            else:
                # Right limit is on wall4
                opp = opp - x
                adj = opp*np.cos(angle)/np.sin(angle)
                intersect_right = 4, y_max - adj
        else:
            # Right limit less than 90 deg., visible may be wall1 or
            # wall2
            angle = np.pi/2 - theta_2
            opp = vertical_dist*np.sin(angle)/np.cos(angle)
            if x + opp <= x_max:
                # Right limit is on wall1
                intersect_right = 1, x + opp
            else:
                # Right limit is on wall2
                opp = x + opp - x_max
                adj = opp*np.cos(angle)/np.sin(angle)
                intersect_right = 2, y_max - adj

    # Robot's orientation angle lies in the third quadrant
    elif theta >= np.pi and theta < 3*np.pi/2:

        if theta_1 == 3*np.pi/2:
            # Left limit is exactly 270 deg., visible is wall3
            intersect_left = 3, x
        elif theta_1 <= 3*np.pi/2:
            # Left limit less than 270 deg., visible may be wall3 and
            # wall4
            angle = 3*np.pi/2 - theta_1
            opp = y*np.sin(angle)/np.cos(angle)
            if x - opp >= 0:
                # Left limit is on wall3
                intersect_left = 3, x - opp
            else:
                # Left limit is on wall4
                opp = opp - x
                adj = opp*np.cos(angle)/np.sin(angle)
                intersect_left = 4, adj
        else:
            # Left limit exceeds 270 deg., visible may be wall2 or
            # wall3
            angle = theta_1 - 3*np.pi/2
            opp = y*np.sin(angle)/np.cos(angle)
            if x + opp <= x_max:
                # Left limit is on wall3
                intersect_left = 3, x + opp
            else:
                # Left limit is on wall2
                opp = x + opp - x_max
                adj = opp*np.cos(angle)/np.sin(angle)
                intersect_left = 2, adj

        if theta_2 == np.pi:
            # Right limit is exactly 180 deg., visible is wall4
            intersect_right = 4, y
        elif theta_2 > np.pi:
            # Right limit exceeds 180 deg., visible may be wall3 or
            # wall4
            angle = 3*np.pi/2 - theta_2
            opp = y*np.sin(angle)/np.cos(angle)
            if x - opp >= 0:
                # Right limit is on wall3
                intersect_right = 3, x - opp
            else:
                # Right limit is on wall4
                opp = opp - x
                adj = opp*np.cos(angle)/np.sin(angle)
                intersect_right = 4, adj
        else:
            # Right limit less than 180 deg., visible may be wall1 or
            # wall4
            angle = np.pi - theta_2
            opp = x*np.sin(angle)/np.cos(angle)
            if y + opp <= y_max:
                # Right limit is on wall4
                intersect_right = 4, y + opp
            else:
                # Right limit is on wall1
                opp = y + opp - y_max
                adj = opp*np.cos(angle)/np.sin(angle)
                intersect_right = 1, adj

    # Robot's orientation angle lies in the 4th quadrant
    elif theta >= 3*np.pi/2:

        if theta_1 == 2*np.pi:
            # Left limit is exactly 360 deg., visible is wall2
            intersect_left = 2, y
        elif theta_1 < 2*np.pi:
            # Left limit less than 360 deg., visible may be wall2 or
            # wall3
            angle = theta_1 - 3*np.pi/2
            opp = y*np.sin(angle)/np.cos(angle)
            if x + opp <= x_max:
                # eft limit on wall3
                intersect_left = 3, x + opp
            else:
                # Left limit on wall2
                opp = x + opp - x_max
                adj = opp*np.cos(angle)/np.sin(angle)
                intersect_left = 2, adj
        else:
            # Left limit exceedds 360 deg., visible may be wall1 or
            # wall2
            angle = theta_1 - 2*np.pi
            opp = horizontal_dist*np.sin(angle)/np.cos(angle)
            if y + opp <= y_max:
                # Left limit is on wall2
                intersect_left = 2, y + opp
            else:
                # Left limit is on wall1
                opp = y + opp - y_max
                adj = opp*np.cos(angle)/np.sin(angle)
                intersect_left = 1, x_max - adj

        if theta_2 == 3*np.pi/2:
            # Right limit is exactly 270 deg., visible is wall3
            intersect_right = 3, x
        elif theta_2 > 3*np.pi/2:
            # Right limit exceeds 270 deg., visible may be wall2 or
            # wall3
            angle = theta_2 - 3*np.pi/2
            opp = y*np.sin(angle)/np.cos(angle)
            if x + opp <= x_max:
                # Right limit is on wall3
                intersect_right = 3, x + opp
            else:
                # Right limit is on wall2
                opp = x + opp - x_max
                adj = opp*np.cos(angle)/np.sin(angle)
                intersect_right = 2, adj
        else:
            # Right limit less than 270 deg., visible may be wall3 or
            # wall4
            angle = 3*np.pi/2 - theta_2
            opp = y*np.sin(angle)/np.cos(angle)
            if x - opp >= 0:
                # Right limit is on wall3
                intersect_right = 3, x - opp
            else:
                # Right limit id on wall4
                opp = opp - x
                adj = opp*np.cos(angle)/np.sin(angle)
                intersect_right = 4, adj

    return intersect_left, intersect_right


def get_angle(adj, opp):
    """
    Calculate triangle angle given two sides.

    @param x: Adjacent side
    @param y: Opposite side
    """

    tan_angle = opp/float(adj)
    angle = np.arctan(tan_angle)

    return angle


def get_walls_view_ratio(il, ir, x, y, theta):
    """
    Get the ratios of the receptors viewed on each wall, in case more
    than one wall is visible. The method returns a list, where the first
    entry indicates how many walls are visible, and each entry following
    indicates the proportion of the wall. Walls are viewed from left to
    right.

    @param il: tuple comprising the number of the wall and the
        coordinate on that wall that represents the left limit of the
        robot's vision range.
    @param ir: tuple comprising the number of the wall and the
        coordinate on that wall that represents the right limit of the
        robot's vision range.
    @param x: Robot's current x position.
    @param y: Robot's current y position.
    @param theta: Robot's current orientation angle.
    """

    wall_left = il[0]
    wall_right = ir[0]
    # View border angles
    theta_1 = theta + np.deg2rad(18)
    theta_2 = abs(theta - np.deg2rad(18))
    theta_1 = theta_1 % (2*np.pi)
    theta_2 = theta_2 % (2*np.pi)
    theta_total = np.deg2rad(36)

    if wall_left == wall_right:
        # Only one wall visible
        return [1]
    elif wall_left - wall_right in (-1, 3):
        # Two walls visible
        if wall_left == 1 and wall_right == 2:
            x_dist = x_max - x
            y_dist = y_max - y
            angle = get_angle(x_dist, y_dist)
            wall_left_proportion = (theta_1 - angle)/theta_total
        elif wall_left == 2 and wall_right == 3:
            x_dist = x_max - x
            y_dist = y
            if theta_1 > 3*np.pi/2:
                angle = 3*np.pi/2 + get_angle(y_dist, x_dist)
                wall_left_proportion = (theta_1 - angle)/theta_total
            else:
                angle = get_angle(x_dist, y_dist)
                wall_left_proportion = (theta_1 + angle)/theta_total
        elif wall_left == 3 and wall_right == 4:
            x_dist = x
            y_dist = y
            angle = np.pi + get_angle(x_dist, y_dist)
            wall_left_proportion = (theta_1 - angle)/theta_total
        elif wall_left == 4 and wall_right == 1:
            x_dist = x
            y_dist = y_max - y
            angle = np.pi/2 + get_angle(y_dist, x_dist)
            wall_left_proportion = (theta_1 - angle)/theta_total

        return [2, wall_left_proportion, 1 - wall_left_proportion]
    else:
        # Three walls visible
        if wall_left == 3 and wall_right == 1:
            x_dist = x
            y_dist = y
            angle1 = get_angle(x_dist, y_dist)
            angle = np.pi + angle1
            wall_left_proportion = (theta_1 - angle)/theta_total
            y_dist = y_max - y
            angle2 = get_angle(x_dist, y_dist)
            wall_middle_proportion = (angle1 + angle2)/theta_total
        else:
            x_dist = x_max - x
            y_dist = y_max - y
            angle1 = get_angle(x_dist, y_dist)
            wall_left_proportion = (theta_1 - angle1)/theta_total
            y_dist = y
            angle2 = get_angle(x_dist, y_dist)
            angle = angle1 + angle2
            wall_middle_proportion = angle/theta_total

        return [3, wall_left_proportion, wall_middle_proportion,
                1 - (wall_left_proportion + wall_middle_proportion)]


def set_wheel_speeds(v_left, v_right, x_cur, y_cur, theta_cur):
    """
    Adjust the calculated wheel speeds from the neural network to the
    actual robot based on boundary conditions.

    @param v_left: Desired left wheel speed.
    @param v_right: Desired right wheel speed.
    @param x_cur: Robot's current x position.
    @param y_cur: Robot's current y position.
    @param theta_cur: Robot's current orientation angle.
    """

    v_t = get_linear_velocity(v_left, v_right)
    w_t = get_angular_velocity(v_left, v_right, v_t)
    collision = False
    x_next, y_next, theta_next = move(x_cur, y_cur, theta_cur, v_t, w_t)
    if x_next < 0 or x_next > x_max:
        x_next = 0 if x_next < 0 else x_max
        v_left= 0
        v_right = 0
        collision = True
    if y_next < 0 or y_next > y_max:
        y_next = 0 if y_next < 0 else y_max
        v_left = 0
        v_right = 0
        collision = False

    return v_left, v_right, collision


def move(x_cur, y_cur, theta_cur, v_t, w_t):
    """
    Update robot's current position and orientation based on the linear
    and angular velocities of the robot, and its current position and
    orientation.

    @param x_cur: robot's current x position.
    @param y_cur: robot's current y position.
    @param theta_cur: robot's current orientation.
    @param v_t: robot's current linear velocity in mm/s.
    @param w_t: robot's current angular velocity in mm/s.
    """

    # Overall robot linear and angular velocities
    theta_dot = w_t
    theta_cur += theta_dot*0.1
    x_dot = v_t*np.cos(theta_cur)#*abs(random.gauss(0, 1))
    y_dot = v_t*np.sin(theta_cur)#*abs(random.gauss(0, 1))

    # Update Current position and orientation
    x_cur += x_dot*0.1
    y_cur += y_dot*0.1

    return x_cur, y_cur, theta_cur


def get_linear_velocity(v_l, v_r):
    """
    Calculate the robot's linear velocity based on left and right
    wheels' speeds.

    @param v_l: left wheel speed in mm/s.
    @param v_r: right wheel speed in mm/s.
    """

    # If both wheels have equal speeds, robot is moving in straight line
    if v_l == v_r:
        v_t = v_l
    # If wheels have exactly opposite speeds, linear velocity is zero.
    elif v_l == -v_r:
        v_t = 0
    else:
        v_t = 0.5*(v_l + v_r)

    return v_t


def get_angular_velocity(v_l, v_r, v_t, L=55):
    """
    Calculate the robot's angular velocity based on left and right
    wheels' speeds.
    
    @param v_l: left wheel speed in mm/s.
    @param v_r: right wheel speed in mm/s.
    @param v_t: robot's current linear velocity in mm/s.
    @param L: distance between both wheels in mm.
    """

    # If robot is moving in a straigh line, angular velocity is zero
    if v_l == v_r:
        w_t = 0
    # If wheels have exactly opposite speeds, robot is moving in
    # circular path with ICC on the mid-point between wheels.
    elif v_l == -v_r:
        w_t = 2*v_r/float(L)
    else:
        # Instantaneous curvature radius of the robot trajectory,
        # relative to the midpoint axis.
        R = (L/2.)*((v_l + v_r)/float(v_l - v_r))

        # Angular velocity of the robot
        w_t = -v_t/float(R)

    return w_t


def fix_boundary_condition(coord):
    """
    """
        
    if coord in (0, 1):
        coord = 1
    else:
        coord -= 1

    return coord


def get_view(x, y, intersect_l, intersect_r, view_proportion):
    """
    Extract the pixel values that the robot sees based on the visible
    wall coordinates. If multiple walls visible, stich values together,
    neglecting projection effect.

    @param x: Robot's current x position.
    @param y: Robot's current y position.
    @param intersect_l: tuple comprising the number of the wall and the
        coordinate on that wall that represents the left limit of the
        robot's vision range.
    @param intersect_r: tuple comprising the number of the wall and the
        coordinate on that wall that represents the right limit of the
        robot's vision range.
    @param view_poportion: The proportion of the view that each wall
                            covers, in case multiple walls visible.
    """

    wall_left = intersect_l[0]
    wall_right = intersect_r[0]

    visible_wall_left = wall_dict[wall_left]
    visible_wall_right = wall_dict[wall_right]

    coordinate_left = int(round(intersect_l[1]))
    coordinate_left = fix_boundary_condition(coordinate_left)

    coordinate_right = int(round(intersect_r[1]))
    coordinate_right = fix_boundary_condition(coordinate_right)

    # In case robot sees only one wall
    if view_proportion[0] == 1:
        # If wall is 2 or 3, left coordinate is greater than right
        # coordinate Slice array from smaller to larger coordinate then
        # reverse
        if coordinate_left == coordinate_right:
             view = [visible_wall_left[coordinate_left]]
        elif wall_left in (2, 3):
            view = visible_wall_left[coordinate_right:coordinate_left]
            view = view[::-1]
        # If wall is 1 or 4, slicing from left to right
        else:
            view = visible_wall_left[coordinate_left:coordinate_right]
        view_tuples = breakdown_view(view)
        view_angles = get_stripes_angle(view_tuples, x, y, coordinate_left,
                                        wall_left)
        if len(view_angles) == 0:
            print x, y, intersect_l, intersect_r, view_proportion
        view = get_photoreceptors_values(view_angles, 64)
    # In case robot sees more than one wall
    elif view_proportion[0] == 2:
        # In case robot sees two walls
        receptors_l = int(np.round(64*view_proportion[1]))
        if receptors_l == 0:
            receptors_l = 1
        elif receptors_l == 64:
            receptors_l = 63
        receptors_r = 64 - receptors_l
        if wall_left in (2, 3):
            total_view_l = visible_wall_left[:coordinate_left]
            if wall_right == 3:
                total_view_r = visible_wall_right[coordinate_right:]
                crl = x_max
                cll = coordinate_left
            else:
                total_view_r = visible_wall_right[:coordinate_right]
                cll = coordinate_left
                crl = 0
        else:
            total_view_l = visible_wall_left[coordinate_left:]
            total_view_r = visible_wall_right[coordinate_right:]
            cll = coordinate_left
            if wall_left == 1:
                crl = y_max
            else:
                crl = 0
        view_l_tuples = breakdown_view(total_view_l)
        view_l_angles = get_stripes_angle(view_l_tuples, x, y, cll, wall_left)
        view_r_tuples = breakdown_view(total_view_r)
        view_r_angles = get_stripes_angle(view_r_tuples, x, y, crl, wall_right)
        if view_l_tuples == [] or view_r_tuples == []:
            print intersect_l, intersect_r, view_proportion
        if len(view_l_angles) == 0 or len(view_r_angles) == 0:
            print x, y, intersect_l, intersect_r, view_proportion
        view_l = get_photoreceptors_values(view_l_angles, receptors_l)
        view_r = get_photoreceptors_values(view_r_angles, receptors_r)
        view = np.concatenate([view_l, view_r])
    else:
        # In case robot sees 3 walls
        receptors_l = int(np.round(64*view_proportion[1]))
        if receptors_l == 0:
            receptors_l = 1
        receptors_r = int(np.round(64*view_proportion[3]))
        if receptors_r == 0:
            receptors_r = 1
        receptors_m = 64 - (receptors_l + receptors_r)
        if wall_left == 1:
            total_view_l = visible_wall_left[coordinate_left:]
            total_view_r = visible_wall_right[coordinate_right:]
            total_view_m = wall_dict[2]
            wall_m = 2
            cll = coordinate_left
            cml = y_max
            crl = x_max
        elif wall_left == 3:
            total_view_l = visible_wall_left[:coordinate_left]
            total_view_r = visible_wall_right[:coordinate_right]
            total_view_m = wall_dict[4]
            wall_m = 4
            cll = coordinate_left
            cml = 0
            crl = 0
        view_l_tuples = breakdown_view(total_view_l)
        view_l_angles = get_stripes_angle(view_l_tuples, x, y, cll,
                                          wall_left)
        view_m_tuples = breakdown_view(total_view_m)
        view_m_angles = get_stripes_angle(view_m_tuples, x, y, cml, wall_m)
        view_r_tuples = breakdown_view(total_view_r)
        view_r_angles = get_stripes_angle(view_r_tuples, x, y, crl, wall_right)
        if view_l_tuples == [] or view_m_tuples == []or view_r_tuples == []:
            print intersect_l, intersect_r, view_proportion
        if len(view_l_angles) == 0 or len(view_m_angles) == 0 or \
                                                len(view_r_angles) == 0:
            print x, y, intersect_l, intersect_r, view_proportion 
        view_l = get_photoreceptors_values(view_l_angles, receptors_l)
        view_m = get_photoreceptors_values(view_m_angles, receptors_m)
        view_r = get_photoreceptors_values(view_r_angles, receptors_r)
        view = np.concatenate([view_l, view_m, view_r])

    return view


def breakdown_view(view):
    """
    Divide view into tuples, one for each stripe, indicating the color
    of the stripe and the length of the stripe in millimeters.

    @param view: Complete view list
    """

    view_tuples = []
    c = 0
    for i in range(len(view)):
        if i == 0:
            c += 1
        elif view[i] == view[i-1]:
            c += 1
        else:
            view_tuples.append((view[i-1], c))
            c = 1

        if i == len(view) - 1:
            view_tuples.append((view[i], c))

    return view_tuples


def get_stripes_angle(view_tuples, x, y, cl, wall):
    """
    Calculate the proportion of the total view angle that each stripe
    covers.

    @param view_tuples: List of tuples indicating the color and length
                        of each stripe in the view.
    @param x: Robot's current x position.
    @param y: Robot's current y position.
    @param cl: the leftmost absolute coordinate in the view.
    @param wall: visible wall number.
    """

    stripes_angles = []
    acc_dist = cl
    for stripe in view_tuples:
        s1 = stripe[1]
        if wall == 1:
            s2 = distance(x, y, acc_dist, y_max)
            s3 = distance(x, y, acc_dist + s1, y_max)
            acc_dist += s1
        elif wall == 2:
            s2 = distance(x, y, x_max, acc_dist)
            s3 = distance(x, y, x_max, acc_dist - s1)
            acc_dist -= s1
        elif wall == 3:
            s2 = distance(x, y, acc_dist, 0)
            s3 = distance(x, y, acc_dist - s1, 0)
            acc_dist -= s1
        else:
            s2 = distance(x, y, 0, acc_dist)
            s3 = distance(x, y, 0, acc_dist + s1)
            acc_dist += s1

        ang = np.arccos((s2**2 + s3**2 - s1**2)/(2*s2*s3))
        stripes_angles.append([stripe[0], ang])

    acc_angles = sum([i[1] for i in stripes_angles])
    for stripe in stripes_angles:
        stripe[1] /= acc_angles

    return stripes_angles


def distance(x1, y1, x2, y2):
    """
    Calculate the distance between two points.

    @param x1: x position of point 1
    @param y1: y position of point 1
    @param x2: x position of point 2
    @param y2: y position of point 2
    """

    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def get_photoreceptors_values(view_angles, n):
    """
    Calculate the values of the robot's photoreceptors, based on the
    view tuples.

    @param view_angles: List of the view view angles for the visible
                        stripes.
    @param n: Number of photoreceptors assigned to that view.
    """

    receptors = []
    for i in range(len(view_angles)):
        value = view_angles[i][0]
        length = int(np.round(view_angles[i][1]*n))
        if length == 0:
            length = 1
        add_item_to_list(receptors, value, length)
    # Because of rounding, the final list's length may not match 64. If
    # this is the case, remove or duplicate random items.
    if len(receptors) == 0:
        print view_angles, n
    while len(receptors) != n:
        rand = np.random.randint(len(receptors))
        if len(receptors) > n:
            del receptors[rand]
        else:
            receptors.insert(rand, receptors[rand])

    return receptors


def get_equally_spaced_pixels(view, num_pxls):
    """
    Get equally spaced pixels from a list of view based on a desired
    number of output pixels.

    @param view: Complete view list.
    @param num_pxls: Desired number of output pixels.
    """

    if len(view) < num_pxls:
        return view
    step = np.round(len(view)/float(num_pxls))
    view_temp = view[::step]
    count = 0
    while len(view_temp) != num_pxls:
        count -= 1
        if len(view_temp) < num_pxls:
            view_temp = np.append(view_temp, [view[count]])
        else:
            view_temp = np.delete(view_temp, count)

    return view_temp


def add_item_to_list(lst, item, n):
    """
    Helper function to append item n times to a list.

    @param lst: The list to which the item will be appended.
    @param item: The item to be appended to the list.
    @param n: Number of times the item should be appended to the list.
    """

    lst.extend([item]*n)
    return lst


def view_image(view):
    """
    Display what the robot sees by stacking several copies of the
    one-row image visible to the robot.

    @param view: pixel values of one row of the robot's view.
    """

    # Stack more copies of the pixel values row for visualization
    # purposes.
    line = view
    for i in range(1000):
        view = np.vstack((view, line))
    print view.shape
    ypixels, xpixels = view.shape
    dpi = 400.
    xinch = xpixels/dpi
    yinch = ypixels/dpi
    #plt.imshow(view, cmap=plt.get_cmap('gray'))
    fig = plt.gcf()
    fig.set_size_inches(xinch, yinch)    
    ax = plt.axes([0., 0., 1., 1.], xticks=[],yticks=[])
    ax.imshow(view, interpolation='none', cmap=plt.get_cmap('gray'))
    plt.show()
    return view

def print_walls():
    """Show all wall images."""

    for i in wall_dict:
        print "Wall %d" % i
        view_image(wall_dict[i])


def same_sign(x, y):
    """
    Helper function to check whether two numbers have the same sign or
    not.

    @param x: First number.
    @param y: Second number.
    """

    return True if (x > 0 and y > 0) or (x < 0 and y < 0) else False


def proximity_to_wall(x, y):
    """
    """

    horizontal_dist = x if x < (x_max - x) else (x_max - x)
    vertical_dist = y if y < (y_max - y) else (y_max - y)

    dist =  horizontal_dist if horizontal_dist < vertical_dist else \
            vertical_dist
    return dist
