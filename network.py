# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:14:02 2015

@author: akl
"""

import numpy as np
import nest
import environment as env
import evolution as ev
import random


def set_receptors_firing_rate(x, y, theta, err_l, err_r):
    """
    Set the firing rate of the 18 neural receptors according to the
    robot's current view.

    @param x: Robot's current x position.
    @param y: Robot's current y position.
    @param theta: Robot's current orientation angle.
    @param err_l: Error between desired and actual speed in the left
                    wheel.
    @param err_r: Error between desired and actual speed in the right
                    wheel.
    """

    il, ir = env.get_visible_wall_coordinates(x, y, theta)
    view_proportion = env.get_walls_view_ratio(il, ir, x, y, theta)
    view = env.get_view(x, y, il, ir, view_proportion)
    if len(view) != 64:
        print len(view), il, ir, view_proportion

    # Input pixels
    px = view[::4]
    while len(px) > 16:
        print len(px)
        px = np.delete(px, -1)
    px = add_noise_to_pixels(px)
    px = list(np.abs(laplace_filter(px)))
    px = scale_list(px)

    simtime = nest.GetKernelStatus()['time']

    # Update the 16 spike generators representing the visual
    # input according to the pixel values read
    for i in range(len(px)):
        if np.random.rand() < px[i]:
            nest.SetStatus([i+11], {'spike_times': [simtime + 1]})

    # Set random rates for the two spike generators representing the
    # error in wheel speeds.
    if np.random.rand() < err_l:
        nest.SetStatus([27],
                       {'spike_times': [simtime + 1]})

    if np.random.rand() < err_r:
        nest.SetStatus([28],
                       {'spike_times': [simtime + 1]})

    return px


def add_noise_to_pixels(p):
    """
    Add uniformly distributed noise to pixel values.

    @param px: List of pixel values.
    """

    for i in range(len(p)):
        noise = random.randint(0, 50)
        p[i] = p[i] + noise if p[i] == 0 else p[i] - noise
    return p


def laplace_filter(px):
    """
    Implementation of the laplace filter as specified in the paper
    -0.5 1 -0.5

    @param px: List of pixel values
    """

    result = []
    for i in range(len(px)):
        if i == 0:
            result.append(px[i] - 0.5*px[i] - 0.5*px[i+1])
        elif i == len(px) - 1:
            result.append(px[i] - 0.5*px[i] - 0.5*px[i-1])
        else:
            result.append(px[i] - 0.5*px[i-1] - 0.5*px[i+1])

    return result


def scale_list(p):
    """
    Scale absolute values of a list in the range [0, 1]

    @param p: input list
    """

    p = list(p)
    for i in range(len(p)):
        p[i] = (p[i])/255.

    return p


def get_neuron_firing_rate(events):
    """
    Calculate the firing rate of the motor neuron measured during the
    previous 20 ms., i.e. the number of spikes that occured in the past
    20 seconds divided by 20.

    @param events: Array of the spike timings of the neurons.
    """

    spikes = 0
    i = -1
    if events.any():
        try:
            while(events[i] > nest.GetKernelStatus()['time'] - 20):
                spikes += 1
                i -= 1
        except IndexError:
            pass
    firing_rate = spikes/20.0

    return firing_rate


def get_motor_neurons_firing_rates(motor_spikes):
    """
    Get the firing rates of the four motor neurons, calculated over the
    last 20 ms.
    """

    spikes_left_fwd = motor_spikes[:1]
    spikes_left_bwd = motor_spikes[1:2]
    spikes_right_fwd = motor_spikes[2:3]
    spikes_right_bwd = motor_spikes[3:]

    # Spike times for the left and right wheel neurons
    left_fwd_events = nest.GetStatus(spikes_left_fwd)[0]['events']['times']
    left_bwd_events = nest.GetStatus(spikes_left_bwd)[0]['events']['times']
    right_fwd_events = nest.GetStatus(spikes_right_fwd)[0]['events']['times']
    right_bwd_events = nest.GetStatus(spikes_right_bwd)[0]['events']['times']

    left_fwd_fr = get_neuron_firing_rate(left_fwd_events)
    left_bwd_fr = get_neuron_firing_rate(left_bwd_events)
    right_fwd_fr = get_neuron_firing_rate(right_fwd_events)
    right_bwd_fr = get_neuron_firing_rate(right_bwd_events)

    firing_rates = (left_fwd_fr, left_bwd_fr, right_fwd_fr, right_bwd_fr)

    return firing_rates


def get_wheel_speeds(motor_firing_rates):
    """
    Get the speed of the left and right wheels based on spiking activity
    in the motor neurons.
    """

    left_fwd_fr = motor_firing_rates[0]
    left_bwd_fr = motor_firing_rates[1]
    right_fwd_fr = motor_firing_rates[2]
    right_bwd_fr = motor_firing_rates[3]

    # Map the firing rate of motor neurons into a maximum speed of
    # 80mm/s And determine the wheel speed using the algebraic sum of
    # the values of both neurons setting the amounts of forward and
    # backward speeds for each wheel.

    speed_left = (left_fwd_fr - left_bwd_fr)*80

    speed_right = (right_fwd_fr - right_bwd_fr)*80

    return speed_left, speed_right
