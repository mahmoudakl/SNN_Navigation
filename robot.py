import vision
import arena

import numpy as np
import matplotlib.pyplot as plt

class robot:
    """Create a two-wheeled robot body"""


    def __init__(self, x_init, y_init, theta_init, max_speed=40.0, t_step=0.1):
        """Create robot with initial pose"""

        self.x_cur = x_init
        self.y_cur = y_init
        self.theta_cur = self.theta_prev = theta_init
        self.max_speed = max_speed
        self.t_step = t_step
        self.arena = arena.arena(1)
        self.intersect_left = ()
        self.intersect_right = ()
        self.get_visible_wall_coordinates()
        self.fig = plt.figure()
        self.replot_robot()


    def replot_robot(self):
        """
        """
        plt.clf()
        plt.xlim(0, self.arena.maximum_width)
        plt.ylim(0, self.arena.maximum_length)
        plt.xticks([])
        plt.yticks([])
        plt.plot(self.x_cur, self.y_cur, marker=(3, 0, np.rad2deg(self.theta_cur) - 90))
        self.fig.show()
    

    def calculate_reward(self, v_left, v_right):
        """
        """
        if v_left < 0 or v_right < 0:
            return 0


    def move(self, v_left, v_right):
        """
        Update robot's current position and orientation based on the linear
        and angular velocities of the robot, and its current position and
        orientation and return the reward based on the motion.

        @param v_left: robot's left wheel speed in mm/s.
        @param v_right: robot's right wheel speed in mm/s.
        """

        v_t = get_linear_velocity(v_left, v_right)
        w_t = get_angular_velocity(v_left, v_right, v_t)

        # Overall robot linear and angular velocities
        theta_dot = w_t
        self.theta_cur += theta_dot*self.t_step
        x_dot = v_t*np.cos(self.theta_cur)
        y_dot = v_t*np.sin(self.theta_cur)
        if self.detect_collision(x_dot, y_dot):
            # collision detected, robot get no reward
            return 0
        # Update Current position and orientation
        self.x_cur += x_dot*self.t_step
        self.y_cur += y_dot*self.t_step
        self.replot_robot()

        return self.calculate_reward(v_left, v_right)


    def detect_collision(self, x_dot, y_dot):
        """
        Detect if collision will occur based on the wheel speeds set.

        @param x_dot: Robot's speed in the x direction
        @param y_dot: Robot's speed in the y direction
        """

        collision = False
        x_next = self.x_cur + x_dot*self.t_step
        y_next = self.y_cur + y_dot*self.t_step

        if x_next <= 0 or x_next >= self.arena.maximum_length:
            collision = True
        elif y_next <= 0 or y_next >= self.arena.maximum_width:
            collision = True

        return collision


    def get_view(self):
        """
        Extract the pixel values that the robot sees based on the visible
        wall coordinates. If multiple walls visible, stich values together,
        neglecting projection effect.
        """

        self.get_visible_wall_coordinates()
        view_proportion = self.get_walls_view_ratio()
        wall_dict = {1: self.arena.wall1, 2: self.arena.wall2, 3: self.arena.wall3,
                    4: self.arena.wall4}
        wall_left = self.intersect_left[0]
        wall_right = self.intersect_right[0]

        visible_wall_left = wall_dict[wall_left]
        visible_wall_right = wall_dict[wall_right]

        coordinate_left = int(round(self.intersect_left[1]))
        coordinate_left = vision.fix_boundary_condition(coordinate_left)

        coordinate_right = int(round(self.intersect_right[1]))
        coordinate_right = vision.fix_boundary_condition(coordinate_right)

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
            view_tuples = vision.breakdown_view(view)
            view_angles = self.get_stripes_angle(view_tuples, coordinate_left, wall_left)
            #if len(view_angles) == 0:
            #    print x, y, intersect_l, intersect_r, view_proportion
            view = vision.get_photoreceptors_values(view_angles, 64)
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
                    crl = self.arena.maximum_length
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
                    crl = self.arena.maximum_width
                else:
                    crl = 0
            view_l_tuples = vision.breakdown_view(total_view_l)
            view_l_angles = self.get_stripes_angle(view_l_tuples, cll, wall_left)
            view_r_tuples = vision.breakdown_view(total_view_r)
            view_r_angles = self.get_stripes_angle(view_r_tuples, crl, wall_right)
            if view_l_tuples == [] or view_r_tuples == []:
                print intersect_l, intersect_r, view_proportion
            if len(view_l_angles) == 0 or len(view_r_angles) == 0:
                print x, y, intersect_l, intersect_r, view_proportion
            view_l = vision.get_photoreceptors_values(view_l_angles, receptors_l)
            view_r = vision.get_photoreceptors_values(view_r_angles, receptors_r)
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
                total_view_m = self.arena.wall2
                wall_m = 2
                cll = coordinate_left
                cml = self.arena.maximum_width
                crl = self.arena.maximum_length
            elif wall_left == 3:
                total_view_l = visible_wall_left[:coordinate_left]
                total_view_r = visible_wall_right[:coordinate_right]
                total_view_m = self.arena.wall4
                wall_m = 4
                cll = coordinate_left
                cml = 0
                crl = 0
            view_l_tuples = vision.breakdown_view(total_view_l)
            view_l_angles = self.get_stripes_angle(view_l_tuples, cll, wall_left)
            view_m_tuples = vision.breakdown_view(total_view_m)
            view_m_angles = self.get_stripes_angle(view_m_tuples, cml, wall_m)
            view_r_tuples = vision.breakdown_view(total_view_r)
            view_r_angles = self.get_stripes_angle(view_r_tuples, crl, wall_right)
            if view_l_tuples == [] or view_m_tuples == []or view_r_tuples == []:
                print intersect_l, intersect_r, view_proportion
            if len(view_l_angles) == 0 or len(view_m_angles) == 0 or \
                                                    len(view_r_angles) == 0:
                print x, y, intersect_l, intersect_r, view_proportion 
            view_l = vision.get_photoreceptors_values(view_l_angles, receptors_l)
            view_m = vision.get_photoreceptors_values(view_m_angles, receptors_m)
            view_r = vision.get_photoreceptors_values(view_r_angles, receptors_r)
            view = np.concatenate([view_l, view_m, view_r])

        return view


    def get_stripes_angle(self, view_tuples, cl, wall):
        """
        Calculate the proportion of the total view angle that each stripe
        covers.

        @param view_tuples: List of tuples indicating the color and length
                            of each stripe in the view.
        @param cl: the leftmost absolute coordinate in the view.
        @param wall: visible wall number.
        """

        stripes_angles = []
        acc_dist = cl
        for stripe in view_tuples:
            s1 = stripe[1]
            if wall == 1:
                s2 = vision.distance(self.x_cur, self.y_cur, acc_dist, self.arena.maximum_width)
                s3 = vision.distance(self.x_cur, self.y_cur, acc_dist + s1, self.arena.maximum_width)
                acc_dist += s1
            elif wall == 2:
                s2 = vision.distance(self.x_cur, self.y_cur, self.arena.maximum_length, acc_dist)
                s3 = vision.distance(self.x_cur, self.y_cur, self.arena.maximum_length, acc_dist - s1)
                acc_dist -= s1
            elif wall == 3:
                s2 = vision.distance(self.x_cur, self.y_cur, acc_dist, 0)
                s3 = vision.distance(self.x_cur, self.y_cur, acc_dist - s1, 0)
                acc_dist -= s1
            else:
                s2 = vision.distance(self.x_cur, self.y_cur, 0, acc_dist)
                s3 = vision.distance(self.x_cur, self.y_cur, 0, acc_dist + s1)
                acc_dist += s1

            ang = np.arccos((s2**2 + s3**2 - s1**2)/(2*s2*s3))
            stripes_angles.append([stripe[0], ang])

        acc_angles = sum([i[1] for i in stripes_angles])
        for stripe in stripes_angles:
            stripe[1] /= acc_angles

        return stripes_angles


    def get_visible_wall_coordinates(self):
        """
        Calculate the visible part of the walls to the robot based on its
        position and orientation and the visual angle of 36 degrees.
        """

        # The two angles reiesenting the vision range
        theta = self.theta_cur % (2*np.pi)
        theta_1 = theta + np.deg2rad(18)
        theta_2 = theta - np.deg2rad(18)
        vertical_dist = self.arena.maximum_width - self.y_cur
        horizontal_dist = self.arena.maximum_length - self.x_cur

        # Robot's orientation angle lies in the first quadrant
        if theta >= 0 and theta < np.pi/2:

            if theta_1 == np.pi/2:
                # Left limit is exactly 90 deg., visible is wall1
                self.intersect_left = 1, self.x_cur
            elif theta_1 < np.pi/2:
                # Left limit less than 90 deg., visible may be wall1 or
                # wall2
                angle = np.pi/2 - theta_1
                opp = vertical_dist*np.sin(angle)/np.cos(angle)
                if opp + self.x_cur <= self.arena.maximum_length:
                    # Left limit is on wall1
                    self.intersect_left = 1, opp + self.x_cur
                else:
                    # Left limit is on wall 2
                    opp = opp + self.x_cur - self.arena.maximum_length
                    adj = opp*np.cos(angle)/np.sin(angle)
                    self.intersect_left = 2, self.arena.maximum_width - adj
            else:
                # Left limit exceeds 90 deg., visible may be wall1 or wall4
                angle = theta_1 - np.pi/2
                opp = vertical_dist*np.sin(angle)/np.cos(angle)
                if self.x_cur - opp >= 0:
                    # Left limit is on wall1
                    self.intersect_left = 1, self.x_cur - opp
                else:
                    # Left limit is on wall4
                    opp = np.abs(self.x_cur - opp)
                    adj = opp*np.cos(angle)/np.sin(angle)
                    self.intersect_left = 4, self.arena.maximum_width - adj

            if theta_2 == 0:
                # Right limit is exactly 0 deg., visible is wall2
                self.intersect_right = 2, self.y_cur
            elif theta_2 > 0:
                # Right limit exceeds 0 deg., visible may be wall1 or
                # wall2
                adj = vertical_dist*np.cos(theta_2)/np.sin(theta_2)
                if adj + self.x_cur <= self.arena.maximum_length:
                    # Right limit is on wall1
                    self.intersect_right = 1, adj + self.x_cur
                else:
                    # Right limit on wall2
                    angle = np.pi/2 - theta_2
                    opp = adj + self.x_cur - self.arena.maximum_length
                    adj = opp*np.cos(angle)/np.sin(angle)
                    self.intersect_right = 2, self.arena.maximum_width - adj
            else:
                # Right limit less than 0 deg., visible may be wall2 or
                # wall3
                angle = np.abs(theta_2)
                opp = horizontal_dist*np.sin(angle)/np.cos(angle)
                if self.y_cur - opp >= 0:
                    # Right limit on wall2
                    self.intersect_right = 2, self.y_cur - opp
                else:
                    # Right limit on wall3
                    opp = opp - self.y_cur
                    adj = opp*np.cos(angle)/np.sin(angle)
                    self.intersect_right = 3, self.arena.maximum_length - adj

        # Robot's orientation angle lies in the second quadrant
        elif theta >= np.pi/2 and theta < np.pi:

            if theta_1 == np.pi:
                # Left limit is exactly 180 deg., visible is wall4
                self.intersect_left = 4, self.y_cur
            elif theta_1 < np.pi:
                # Left limit less than 180 deg., visible may be wall1 or
                # wall4
                angle = np.pi - theta_1
                adj = vertical_dist*np.cos(angle)/np.sin(angle)
                if self.x_cur - adj >= 0:
                    # Left limit is on wall1
                    self.intersect_left = 1, self.x_cur - adj
                else:
                    # Left limit is on wall2
                    adj = np.abs(self.x_cur - adj)
                    opp = adj*np.sin(angle)/np.cos(angle)
                    self.intersect_left = 4, self.arena.maximum_width - opp
            else:
                # Left limit excceeds 180 deg., visible may be wall3 or
                # wall4
                angle = theta_1 - np.pi
                opp = self.x_cur*np.sin(angle)/np.cos(angle)
                if self.y_cur - opp >= 0:
                    # Left limit on wall4
                    self.intersect_left = 4, self.y_cur - opp
                else:
                    # Left limit on wall3
                    opp = opp - self.y_cur
                    adj = opp*np.cos(angle)/np.sin(angle)
                    self.intersect_left = 3, adj

            if theta_2 == np.pi/2:
                # Right limit is exact;y 90 deg., visible is wall1
                self.intersect_right = 1, self.x_cur
            elif theta_2 > np.pi/2:
                # Right limit more than 90 deg., visible my be wall1 or
                # wall4
                angle = theta_2 - np.pi/2
                opp = vertical_dist*np.sin(angle)/np.cos(angle)
                if self.x_cur - opp >= 0:
                    # Right limit is on wall1
                    self.intersect_right = 1, self.x_cur - opp
                else:
                    # Right limit is on wall4
                    opp = opp - self.x_cur
                    adj = opp*np.cos(angle)/np.sin(angle)
                    self.intersect_right = 4, self.arena.maximum_width - adj
            else:
                # Right limit less than 90 deg., visible may be wall1 or
                # wall2
                angle = np.pi/2 - theta_2
                opp = vertical_dist*np.sin(angle)/np.cos(angle)
                if self.x_cur + opp <= self.arena.maximum_length:
                    # Right limit is on wall1
                    self.intersect_right = 1, self.x_cur + opp
                else:
                    # Right limit is on wall2
                    opp = self.x_cur + opp - self.arena.maximum_length
                    adj = opp*np.cos(angle)/np.sin(angle)
                    self.intersect_right = 2, self.arena.maximum_width - adj

        # Robot's orientation angle lies in the third quadrant
        elif theta >= np.pi and theta < 3*np.pi/2:

            if theta_1 == 3*np.pi/2:
                # Left limit is exactly 270 deg., visible is wall3
                self.intersect_left = 3, self.x_cur
            elif theta_1 <= 3*np.pi/2:
                # Left limit less than 270 deg., visible may be wall3 and
                # wall4
                angle = 3*np.pi/2 - theta_1
                opp = self.y_cur*np.sin(angle)/np.cos(angle)
                if self.x_cur - opp >= 0:
                    # Left limit is on wall3
                    self.intersect_left = 3, self.x_cur - opp
                else:
                    # Left limit is on wall4
                    opp = opp - self.x_cur
                    adj = opp*np.cos(angle)/np.sin(angle)
                    self.intersect_left = 4, adj
            else:
                # Left limit exceeds 270 deg., visible may be wall2 or
                # wall3
                angle = theta_1 - 3*np.pi/2
                opp = self.y_cur*np.sin(angle)/np.cos(angle)
                if self.x_cur + opp <= self.arena.maximum_length:
                    # Left limit is on wall3
                    self.intersect_left = 3, self.x_cur + opp
                else:
                    # Left limit is on wall2
                    opp = self.x_cur + opp - self.arena.maximum_length
                    adj = opp*np.cos(angle)/np.sin(angle)
                    self.intersect_left = 2, adj

            if theta_2 == np.pi:
                # Right limit is exactly 180 deg., visible is wall4
                self.intersect_right = 4, self.y_cur
            elif theta_2 > np.pi:
                # Right limit exceeds 180 deg., visible may be wall3 or
                # wall4
                angle = 3*np.pi/2 - theta_2
                opp = self.y_cur*np.sin(angle)/np.cos(angle)
                if self.x_cur - opp >= 0:
                    # Right limit is on wall3
                    self.intersect_right = 3, self.x_cur - opp
                else:
                    # Right limit is on wall4
                    opp = opp - self.x_cur
                    adj = opp*np.cos(angle)/np.sin(angle)
                    self.intersect_right = 4, adj
            else:
                # Right limit less than 180 deg., visible may be wall1 or
                # wall4
                angle = np.pi - theta_2
                opp = self.x_cur*np.sin(angle)/np.cos(angle)
                if self.y_cur + opp <= self.arena.maximum_width:
                    # Right limit is on wall4
                    intersect_right = 4, self.y_cur + opp
                else:
                    # Right limit is on wall1
                    opp = self.y_cur + opp - arena.maximum_width
                    adj = opp*np.cos(angle)/np.sin(angle)
                    self.intersect_right = 1, adj

        # Robot's orientation angle lies in the 4th quadrant
        elif theta >= 3*np.pi/2:

            if theta_1 == 2*np.pi:
                # Left limit is exactly 360 deg., visible is wall2
                self.intersect_left = 2, self.y_cur
            elif theta_1 < 2*np.pi:
                # Left limit less than 360 deg., visible may be wall2 or
                # wall3
                angle = theta_1 - 3*np.pi/2
                opp = self.y_cur*np.sin(angle)/np.cos(angle)
                if self.x_cur + opp <= self.arena.maximum_length:
                    # eft limit on wall3
                    self.intersect_left = 3, self.x_cur + opp
                else:
                    # Left limit on wall2
                    opp = self.x_cur + opp - self.arena.maximum_length
                    adj = opp*np.cos(angle)/np.sin(angle)
                    self.intersect_left = 2, adj
            else:
                # Left limit exceedds 360 deg., visible may be wall1 or
                # wall2
                angle = theta_1 - 2*np.pi
                opp = horizontal_dist*np.sin(angle)/np.cos(angle)
                if self.y_cur + opp <= self.arena.maximum_width:
                    # Left limit is on wall2
                    self.intersect_left = 2, self.y_cur + opp
                else:
                    # Left limit is on wall1
                    opp = self.y_cur + opp - self.arena.maximum_width
                    adj = opp*np.cos(angle)/np.sin(angle)
                    self.intersect_left = 1, self.arena.maximum_length - adj

            if theta_2 == 3*np.pi/2:
                # Right limit is exactly 270 deg., visible is wall3
                self.intersect_right = 3, self.x_cur
            elif theta_2 > 3*np.pi/2:
                # Right limit exceeds 270 deg., visible may be wall2 or
                # wall3
                angle = theta_2 - 3*np.pi/2
                opp = self.y_cur*np.sin(angle)/np.cos(angle)
                if self.x_cur + opp <= self.arena.maximum_length:
                    # Right limit is on wall3
                    self.intersect_right = 3, self.x_cur + opp
                else:
                    # Right limit is on wall2
                    opp = self.x_cur + opp - self.arena.maximum_length
                    adj = opp*np.cos(angle)/np.sin(angle)
                    self.intersect_right = 2, adj
            else:
                # Right limit less than 270 deg., visible may be wall3 or
                # wall4
                angle = 3*np.pi/2 - theta_2
                opp = self.y_cur*np.sin(angle)/np.cos(angle)
                if self.x_cur - opp >= 0:
                    # Right limit is on wall3
                    self.intersect_right = 3, self.x_cur - opp
                else:
                    # Right limit id on wall4
                    opp = opp - self.x_cur
                    adj = opp*np.cos(angle)/np.sin(angle)
                    self.intersect_right = 4, adj


    def get_walls_view_ratio(self):
        """
        Get the ratios of the receptors viewed on each wall, in case more
        than one wall is visible. The method returns a list, where the first
        entry indicates how many walls are visible, and each entry following
        indicates the proportion of the wall. Walls are viewed from left to
        right.
        """

        wall_left = self.intersect_left[0]
        wall_right = self.intersect_right[0]

        # View border angles
        theta_1 = self.theta_cur + np.deg2rad(18)
        theta_2 = abs(self.theta_cur - np.deg2rad(18))
        theta_1 = theta_1 % (2*np.pi)
        theta_2 = theta_2 % (2*np.pi)
        theta_total = np.deg2rad(36)

        if wall_left == wall_right:
            # Only one wall visible
            return [1]
        elif wall_left - wall_right in (-1, 3):
            # Two walls visible
            if wall_left == 1 and wall_right == 2:
                x_dist = self.arena.maximum_length - self.x_cur
                y_dist = self.arena.maximum_width - self.y_cur
                angle = vision.get_angle(x_dist, y_dist)
                wall_left_proportion = (theta_1 - angle)/theta_total
            elif wall_left == 2 and wall_right == 3:
                x_dist = self.arena.maximum_length - self.x_cur
                y_dist = self.y_cur
                if theta_1 > 3*np.pi/2:
                    angle = 3*np.pi/2 + vision.get_angle(y_dist, x_dist)
                    wall_left_proportion = (theta_1 - angle)/theta_total
                else:
                    angle = vision.get_angle(x_dist, y_dist)
                    wall_left_proportion = (theta_1 + angle)/theta_total
            elif wall_left == 3 and wall_right == 4:
                x_dist = self.x_cur
                y_dist = self.y_cur
                angle = np.pi + vision.get_angle(x_dist, y_dist)
                wall_left_proportion = (theta_1 - angle)/theta_total
            elif wall_left == 4 and wall_right == 1:
                x_dist = self.x_cur
                y_dist = self.arena.maximum_width - self.y_cur
                angle = np.pi/2 + vision.get_angle(y_dist, x_dist)
                wall_left_proportion = (theta_1 - angle)/theta_total

            return [2, wall_left_proportion, 1 - wall_left_proportion]
        else:
            # Three walls visible
            if wall_left == 3 and wall_right == 1:
                x_dist = self.x_cur
                y_dist = self.y_cur
                angle1 = vision.get_angle(x_dist, y_dist)
                angle = np.pi + angle1
                wall_left_proportion = (theta_1 - angle)/theta_total
                y_dist = self.arena.maximum_width - self.y_cur
                angle2 = vision.get_angle(x_dist, y_dist)
                wall_middle_proportion = (angle1 + angle2)/theta_total
            else:
                x_dist = self.arena.maximum_length - self.x_cur
                y_dist = self.arena.maximum_width - self.y_cur
                angle1 = vision.get_angle(x_dist, y_dist)
                wall_left_proportion = (theta_1 - angle1)/theta_total
                y_dist = self.y_cur
                angle2 = vision.get_angle(x_dist, y_dist)
                angle = angle1 + angle2
                wall_middle_proportion = angle/theta_total

            return [3, wall_left_proportion, wall_middle_proportion,
                    1 - (wall_left_proportion + wall_middle_proportion)]


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
