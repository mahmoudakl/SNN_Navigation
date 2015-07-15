# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:14:02 2015

@author: akl

This module is responsible for building a Spiking Neural Network with
NEST according to an evolutionary  approach and a learning approach.
It also includes function that extract and update network data, e.g.
spike times and firing rates.
"""
import numpy as np
import random

import nest

import environment as env

neurons, receptors = [], []
receptor_spikes, neurons_spikes, motor_spikes, voltmeters = [], [], [], []


def create_evolutionary_network(genetic_string, model):
    """
    Create neural network based on the encoded topology in the genetic
    string.

    @param genetic_string: Binary string encoding synaptic connections.
    @param neuronal_model: Neuronal model for the 10 neurons.
    """

    global neurons, receptors, voltmeters, receptor_spikes,\
    neurons_spikes, motor_spikes

    if model == 'mat':
        neuron_params = {'E_L': 0.0, 'V_m': 0.0, 'tau_m': 4.0, 'C_m': 10.0,
                         'tau_syn_ex': 3.0, 'tau_syn_in': 3.0, 'omega': 0.1,
                         'alpha_1': 1.0, 'alpha_2': 0.0,
                         't_ref': 0.1, 'tau_1': 4.0}
        # 10 mat neurons
        neurons = nest.Create('mat2_psc_exp', 10, neuron_params)
    else:
        neuron_params = {'V_m': 0.0, 'E_L': 0.0, 'C_m': 10.0, 'tau_m': 4.0,
                         't_ref': 1.0, 'V_th': 0.1,
                         'V_reset': -0.1, 'tau_syn': 1.0}
        neurons = nest.Create('iaf_neuron', 10, neuron_params)

    # The last 4 neurons are the ones used to set wheel speeds'
    motor_neurons = neurons[6:]

    # 18 poisson generators representing neural receptors
    receptors = nest.Create('spike_generator', 18)

    # Spike detetctors for receptors and neurons for testing purposes.
    population_spikes = nest.Create('spike_detector', 2,
                                    [{'label': 'receptors'},
                                     {'label': 'neurons'}])
    receptor_spikes = population_spikes[:1]
    neurons_spikes = population_spikes[1:]

    # Four spike detectors for the motor neurons
    motor_spikes = nest.Create('spike_detector', 4,
                                        [{'label': 'left_wheel_forward'},
                                        {'label': 'left_wheel_backward'},
                                        {'label': 'right_wheel_forward'},
                                        {'label': 'right_wheel_backward'}])

    # Four voltmeters to plot motor neuron's voltage trace
    voltmeters = nest.Create('voltmeter', 4, {'withgid': True})

    # Setting the excitatory synapse model parameters
    nest.CopyModel('static_synapse', 'e', {'weight': 1.0, 'delay': 2.0})
    # Setting the inhibitory synapse model parameters
    nest.CopyModel('static_synapse', 'i', {'weight': -1.0, 'delay': 2.0})

    # Connect spike detectors to neurons and neural receptors
    nest.Connect(receptors, receptor_spikes, syn_spec='e')
    nest.Connect(neurons, neurons_spikes)

    # Connect spike detectors and voltmeters to motor neurons
    for i in range(len(motor_neurons)):
        nest.Connect(motor_neurons[i], motor_spikes[i])
        nest.Connect(voltmeters[i], motor_neurons[i])

    # Set Synaptic properties of the 10 neurons
    neuron_synapses = genetic_string[:, 0]

    # Establish Connections between neurons according to genetic string
    for target in range(len(genetic_string)):
        for source in range(1, 11):
            if genetic_string[target][source]:
                synapse = ('e' if neuron_synapses[source-1] else 'i')
                nest.Connect(neurons[source-1], neurons[target],
                             syn_spec=synapse)

    # Establish Connections between receptors and neurons based on
    # genetic string.
        for source in range(11, 29):
            if genetic_string[target][source]:
                nest.Connect(receptors[source-11], neurons[target],
                             syn_spec='e')
    return motor_spikes


def set_receptors_firing_rate(x, y, theta, err_l, err_r):
    """
    Set the firing rate of the 18 neural receptors according to the
    robot's current view and the error in wheel speeds.

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


def update_refratory_perioud(model):
    """
    Multiply the constant value of the refractory period by a uniformly
    random variable in the range [0, 1].
    """

    if model == 'mat':
        for i in range(1, 11):
            nest.SetStatus([i], {'alpha_1': np.random.uniform(0, 1),
                                 't_ref': np.random.uniform(0.1, 1)})
    else:
        for i in range(1, 11):
            nest.SetStatus([i], {'V_reset': np.random.uniform(0, -0.1),
                                 't_ref': np.random.uniform(0.1, 1)})


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


def get_average_spikes_per_second():
    """
    Get the avergae number of spikes per seond for each one of the 10
    neurons from the spike detector data
    """

    spikes = []
    spike_senders = nest.GetStatus(neurons_spikes, 'events')[0]['senders']
    for i in range(1, 11):
        spikes.append(len(spike_senders[spike_senders == i])/40)

    return spikes


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


def get_voltmeter_data():
    """
    Extract data from voltmeters.
    """

    voltmeter_data = []
    for v in voltmeters:
        vm = nest.GetStatus([v], 'events')[0]
        V, t = vm['V_m'], vm['times']
        voltmeter_data.append((V, t))

    return voltmeter_data


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
