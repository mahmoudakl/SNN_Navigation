# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:07:05 2015

@author: akl
"""

import numpy as np
import nest

pop_id = 1

neurons, receptors = [], []
receptor_spikes, neurons_spikes, motor_spikes, voltmeters = [], [], [], []

model = ''


def generate_population():
    """
    Generate Initial population of 60 individuals.
    """

    population = np.random.randint(2, size=(60, 10, 29))

    return population


def load_population():
    """
    Load population from file and indicate number of available
    individuals.
    """

    population = np.load('populations/population%d.npz' % pop_id)['arr_0'][0]
    num_individulas = len(population)

    return population, num_individulas


def create_network(genetic_string, neuronal_model):
    """
    Create neural network based on the encoded topology in the genetic
    string.

    @param genetic_string: Binary string encoding synaptic connections.
    @param neuronal_model: Neuronal model for the 10 neurons.
    """

    global neurons, receptors, voltmeters, receptor_spikes,\
    neurons_spikes, motor_spikes, model

    model = neuronal_model
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


def update_refratory_perioud(model='mat'):
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


def get_fitness_value(speed_log):
    """
    Return fitness value of the current genetic string represting a
    network topology, based on the fitness function described in the
    paper.

    @param speed_log: tuple comprising left wheel and right wheel
                        speeds' recorded during simulation.
    """

    phi = 0
    for i in range(len(speed_log)):
        # Fitness functionis zero whenever v_left or v_right are less
        # than zero
        if speed_log[i][0] > 0 and speed_log[i][1] > 0:
            phi += (speed_log[i][0] + speed_log[i][1])
    phi = phi/(len(speed_log)*160.)

    return phi


def get_top_performers(generation_log, num_performers=15):
    """
    Extract the indices of the top individuals from the fitness log.

    @param generation_log: fitness function scores for all individuals
                           in a population.
    @param num_performers: number of top performers to look for. Default
                            value is 15, which corresponds to a
                            truncation threshold of 25%.
    """

    top_performers = []
    fitness_log = [j[1] for j in generation_log]
    for i in range(num_performers):
        max_index = np.argmax(fitness_log)
        maximum = generation_log[max_index]
        top_performers.append(maximum)
        fitness_log[max_index] = -1.0

    return top_performers


def evolve_new_generation(top_performers):
    """
    Evolve a new generation of 60 individuals based on the top 15
    performers from the previous generation, using rank-based truncated
    selection, one-point crossover, bit mutation, and elitism.

    @param top_performers: The top 15 performing individuals from the
                            previous generation.
    """

    population = np.zeros_like(load_population()[0])
    j = 0
    # Produce four copies of each winning individual
    for i in range(1, 61):
        population[i-1] = top_performers[j][0]
        if i % 4 == 0:
            j += 1
    pairs = get_unique_pairs(population)
    for i in pairs:
        if np.random.rand() < 0.1:
            parent1 = population[i[0]]
            parent2 = population[i[1] - 1]
            child1, child2 = one_point_crossover(parent1, parent2)
            population[i[0]] = child1
            population[i[1]] = child2

    # Apply bit mutation for each individual w/ probability 0.05 per bit
    population = bit_mutation(population)

    # Apply Elitism
    rand = np.random.randint(len(population))
    population[rand] = top_performers[0][0]

    return population


def get_unique_pairs(population):
    """
    List the indices all unique pairs in the given list. If the list
    consists of 15 individuals, the number of unique pairs is 105,
    according to (n choose k).

    @param population: List of individuals.
    """

    pairs = []
    for i in range(0, len(population)):
        for j in range(i + 1, len(population)):
            pairs.append((i, j))

    return pairs


def one_point_crossover(parent1, parent2):
    """
    Produce offspring from two parents based on one-point crossover. The
    point is chosen randomly.

    @param parent1: The first parent.
    @param parent2: The second parent.
    """

    parent1 = parent1.reshape((290))
    parent2 = parent2.reshape((290))
    child1 = np.zeros(290, dtype=int)
    child2 = np.zeros(290, dtype=int)
    point = np.random.randint(len(parent1))
    for i in range(point):
        child1[i] = parent1[i]
        child2[i] = parent2[i]
    for i in range(point, 290):
        child1[i] = parent2[i]
        child2[i] = parent1[i]
    child1 = child1.reshape((10, 29))
    child2 = child2.reshape((10, 29))

    return child1, child2


def bit_mutation(population):
    """
    Mutate each individual by changing the value of bit with probability
    0.05 per bit.

    @population: List of individuals.
    """

    for individual in population:
        individual = individual.reshape((290))
        for j in range(290):
            if np.random.rand() < 0.05:
                individual[j] = 0 if individual[j] else 1

    return population


def get_connectivity(individual):
    """
    Calculate the fraction of encoded connections to all possible
    connections.

    @param individual: Binary string encoding synaptic connections.
    """

    count = 0
    for item in individual:
        for i in range(1, len(item)):
            if item[i]:
                count += 1

    return count/float(280)
