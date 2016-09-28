import vision
import arena

import numpy as np


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


    def move(self, v_left, v_right):
        """
        Update robot's current position and orientation based on the linear
        and angular velocities of the robot, and its current position and
        orientation and return the reward based on the motion.

        @param v_t: robot's current linear velocity in mm/s.
        @param w_t: robot's current angular velocity in mm/s.
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
        return x_dot + y_dot


    def detect_collision(self, x_dot, y_dot):
        """
        Detect if collision will occur based on the wheel speeds set.

        @param x_dot: Robot's speed in the x direction
        @param y_dot: Robot's speed in the y direction
        @param arena: The arena in which the robot is moving
        """

        collision = False
        x_next = self.x_cur + x_dot*self.t_step
        y_next = self.y_cur + y_dot*self.t_step

        if x_next <= 0 or x_next >= self.arena.maximum_length:
            collision = True
        elif y_next <= 0 or y_next >= self.arena.maximum_width:
            collision = True

        return collision


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