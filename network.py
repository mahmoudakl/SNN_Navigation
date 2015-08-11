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

import nest

import vision

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

    neurons, receptors = create_nodes(model)

    # The last 4 neurons are the ones used to set wheel speeds'
    motor_neurons = neurons[6:]

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


def create_learning_network(model, w_low, w_high):
    """
    Create NEST Network with 18 receptors, 10 neurons and spik
    detectors. Establish connections with random initial weights between
    all nodes.

    @param model: Neuronal model.
    @param w_low: Low weight range value.
    @param w_high: High weight range value.
    """

    neurons, receptors = create_nodes(model)

    neuron_spike_detectors = nest.Create('spike_detector', 10)
    receptor_spike_detectors = nest.Create('spike_detector', 18)

    population_spikes = nest.Create('spike_detector', 2,
                                    [{'label': 'receptors'},
                                     {'label': 'neurons'}])

    nest.CopyModel('static_synapse', 'e', {'delay': 2.0, 'weight': 1.0})
    nest.CopyModel('static_synapse', 'syn', {'delay': 2.0})

    nest.Connect(receptors, population_spikes[:1], syn_spec='syn')
    nest.Connect(neurons, population_spikes[1:])

    for i in range(len(neurons)):
        nest.Connect(neurons[i], neuron_spike_detectors[i])
    for i in range(len(receptors)):
        nest.Connect(receptors[i], receptor_spike_detectors[i], 'syn')

    nest.Connect(receptors, neurons, {'rule': 'all_to_all'},
                 {'model': 'syn', 'weight': {'distribution': 'uniform',
                                             'low': 0.0, 'high': w_high}})
    nest.Connect(neurons, neurons, {'rule': 'all_to_all'},
                 {'model': 'syn', 'weight': {'distribution': 'uniform',
                                             'low': w_low, 'high': w_high}})

    return neurons, neuron_spike_detectors, receptors,\
            receptor_spike_detectors, population_spikes


def create_nodes(model):
    """
    Create 18 receptors and 10 neurons based on the neuronal model.

    @param model: Neuronal model.
    """

    # Multi-timescale Adaptive Threhsold Neuronal Model
    if model == 'mat':
        neuron_params = {'E_L': 0.0, 'V_m': 0.0, 'tau_m': 4.0, 'C_m': 10.0,
                         'tau_syn_ex': 3.0, 'tau_syn_in': 3.0, 'omega': 0.1,
                         'alpha_1': 1.0, 'alpha_2': 0.0,
                         't_ref': 0.1, 'tau_1': 4.0}
        # 10 mat neurons
        neurons = nest.Create('mat2_psc_exp', 10, neuron_params)
    # Leaky Integrate and Fire Neuronal Model
    else:
        neuron_params = {'V_m': 0.0, 'E_L': 0.0, 'C_m': 50.0, 'tau_m': 4.0,
                         't_ref': np.random.uniform(0.1, 2), 'V_th': 0.1,
                         'V_reset': np.random.uniform(0, -1), 'tau_syn': 10.0}
        # 10 lif neurons
        neurons = nest.Create('iaf_neuron', 10, neuron_params)

    # 18 poisson generators representing neural receptors
    receptors = nest.Create('spike_generator', 18)

    return neurons, receptors


def set_receptors_firing_rate(x, y, theta, err_l, err_r, arena):
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

    il, ir = vision.get_visible_wall_coordinates(x, y, theta, arena)
    view_proportion = vision.get_walls_view_ratio(il, ir, x, y, theta, arena)
    view = vision.get_view(x, y, il, ir, view_proportion, arena)
    if len(view) != 64:
        print len(view), il, ir, view_proportion

    # Input pixels
    px = view[::4]
    while len(px) > 16:
        print len(px)
        px = np.delete(px, -1)
    px = vision.add_noise_to_pixels(px)
    px = list(np.abs(vision.laplace_filter(px)))
    px = vision.scale_list(px)

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
    Add noise to the refractory function of the corresponding neuronal
    model used in the network.

    @param model: Neuronal model used in the network.
    """

    if model == 'mat':
        for i in range(1, 11):
            nest.SetStatus([i], {'alpha_1': np.random.uniform(0, 1),
                                 't_ref': np.random.uniform(0.1, 1)})
    else:
        for i in range(1, 11):
            nest.SetStatus([i], {'V_reset': np.random.uniform(0, -0.1),
                                 't_ref': np.random.uniform(0.1, 1)})


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

    @param motor_spikes: List of global identifiers of spike detectors
                        connected to the motor neurons.
    """

    left_fwd_fr = get_neuron_firing_rate(nest.GetStatus(motor_spikes[:1])[0]\
                                                        ['events']['times'])
    left_bwd_fr = get_neuron_firing_rate(nest.GetStatus(motor_spikes[1:2])[0]\
                                                        ['events']['times'])
    right_fwd_fr = get_neuron_firing_rate(nest.GetStatus(motor_spikes[2:3])[0]\
                                                        ['events']['times'])
    right_bwd_fr = get_neuron_firing_rate(nest.GetStatus(motor_spikes[3:])[0]\
                                                        ['events']['times'])

    return (left_fwd_fr, left_bwd_fr, right_fwd_fr, right_bwd_fr)


def get_voltmeter_data():
    """Extract data from voltmeters."""

    voltmeter_data = []
    for v in voltmeters:
        vm = nest.GetStatus([v], 'events')[0]
        V, t = vm['V_m'], vm['times']
        voltmeter_data.append((V, t))

    return voltmeter_data


def get_wheel_speeds(motor_firing_rates):
    """
    Get the speed of the left and right wheels based on the firing rates
    of the motor neurons. The total wheel speed is the algebraic sum of
    the firing rates of the forward and backward neurons. The maximum
    speed is 80 mm/s.

    @param motor_firing_rates: Firing rates of the motor neurons.
    """

    speed_left = (motor_firing_rates[0] - motor_firing_rates[1])*80
    speed_right = (motor_firing_rates[2] - motor_firing_rates[3])*80

    return speed_left, speed_right
