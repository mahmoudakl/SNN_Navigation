# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:16:59 2015

@author: akl
"""
import numpy as np

import network


def detect_collision(v_left, v_right, x_cur, y_cur, theta_cur, t_step, arena):
    """
    Detect if collision will occur based on the wheel speeds set.

    @param v_left: Desired left wheel speed.
    @param v_right: Desired right wheel speed.
    @param x_cur: Robot's current x position.
    @param y_cur: Robot's current y position.
    @param theta_cur: Robot's current orientation angle.
    @param t_step: Simulation time-step.
    @param arena: Arena object.
    """

    v_t = get_linear_velocity(v_left, v_right)
    w_t = get_angular_velocity(v_left, v_right, v_t)
    collision = False
    x_next, y_next, theta_next = move(x_cur, y_cur, theta_cur, v_t, w_t,
                                      t_step/1000.)
    if x_next <= 0 or x_next >= arena.maximum_length():
        collision = True
    elif y_next <= 0 or y_next >= arena.maximum_width():
        collision = True

    return collision


def update_wheel_speeds(x_cur, y_cur, theta_cur, motor_firing_rates, t_step,
                        arena):
    """
    Update the left and right wheel speeds and calculate the linear and
    angular speeds of the robot.

    @param x_cur: Robot's current x position.
    @param y_cur: Robot's current y position.
    @param theta_cur: Robot's current orientation.
    @param motor_firing_rates: Firing rates of the motor neurons.
    @param arena: Arena object.
    """

    v_l, v_r = network.get_wheel_speeds(motor_firing_rates)
    r1, r2 = np.random.uniform(0.7, 1, 2)
    # get actual speeds by multiplyin the desired speeds with uniform
    # random variables
    v_l_act = v_l*r1
    v_r_act = v_r*r2
    collision = detect_collision(v_l_act, v_r_act, x_cur,  y_cur, theta_cur,
                                 t_step, arena)
    if collision:
        return True, {}
        
    v_t = get_linear_velocity(v_l_act, v_r_act)
    w_t = get_angular_velocity(v_l_act, v_r_act, v_t)

    motion_dict = {'actual_left': v_l_act, 'actual_right': v_r_act,
                   'linear': v_t, 'angular': w_t, 'error_left': 1 - r1,
                   'error_right': 1 - r2, 'desired_left': v_l,
                   'desired_right': v_r}

    return False, motion_dict


def move(x_cur, y_cur, theta_cur, v_t, w_t, t_step):
    """
    Update robot's current position and orientation based on the linear
    and angular velocities of the robot, and its current position and
    orientation.

    @param x_cur: robot's current x position.
    @param y_cur: robot's current y position.
    @param theta_cur: robot's current orientation.
    @param v_t: robot's current linear velocity in mm/s.
    @param w_t: robot's current angular velocity in mm/s.
    @param t_step: Simulation time-step.
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
