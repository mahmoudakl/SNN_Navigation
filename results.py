# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:00:27 2015

@author: akl
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import numpy as np
import os
import environment as env
import evolution as ev
import nest
import nest.voltage_trace
import nest.raster_plot
import pandas as pd

path = ''
population_path = ''
generation_path = ''
plt.rcParams['figure.figsize'] = (12, 6)


def set_results_path(init, model, arena, pop_id):
    """
    Make new directory to save results in, and set the path variable.

    @param init: Variable indicating fixed or random initialization.
    @param model: Neuronal model used in the experiment.
    @param arena: Arena used in the experiment.
    @param pop_id: Population id used in the experiment.s
    """

    global path
    path = '/home/akl/TUM/Masterarbeit/Results/'
    foldername = 'arena%d_population%d_%s_%s' % (arena, pop_id, model, init)
    path += foldername
    os.mkdir(path)


def create_population_folder(p):
    """
    Create a folder for the population to save results in.

    @param p: Population number.
    """

    global path, population_path
    population_path = path + '/population%d' % (p + 1)
    os.mkdir(population_path)


def create_generation_folder(g):
    """
    Create a folder for the generation to save results in.

    @param g: Generation number.
    """

    global path, generation_path
    generation_path = path + '/gen%d' % (g + 1)
    os.mkdir(generation_path)


def create_individual_folder(i):
    """
    Create a folder for the individual to save results in.

    @param i: Individual number.
    """

    global generation_path
    individual_path = generation_path + '/individual%d' % (i + 1)
    os.mkdir(individual_path)


def update_best_individual(best):
    """
    Rename the folder of the best individual inside the generation
    folder to be distinctive.

    @param best: The best individual of the generation.
    """

    print best
    global generation_path
    index = best[2]
    folder = generation_path + '/individual%d' % (index + 1)
    os.rename(folder, generation_path + '/best')


def plot_trajectory(traj, x_init, y_init):
    """
    Plot and save the robot's trajectory during the simulation.

    @param traj: The recorded trajectory points from simulation.
    @param x_init: Initial x position.
    @param y_init: Initial y position.
    """

    global path
    x = [i[0] for i in traj]
    y = [i[1] for i in traj]

    plt.rcParams['figure.figsize'] = (10, 6)
    plt.figure('Trajectory')
    plt.subplot(1, 1, 1)
    plt.xlim(0, env.x_max)
    plt.ylim(0, env.y_max)
    plt.xticks([])
    plt.yticks([])

    plt.plot(x_init, y_init, 'Dr')
    plt.plot(x, y, color="Black", linewidth=1.0)


def plot_average_fitness_evolution(average_fitness):
    """
    Plot fitness averaged over all population individuals for all evoled
    generations.

    @param average_fitness: Average fitness per generation for each
                            population.
    """

    global path
    generations = range(1, len(average_fitness) + 1)

    plt.figure()
    plt.subplot(1, 1, 1)
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    colors = ['r', 'b', 'y', 'g', ]

    for i in range(len(average_fitness)):
        plt.plot(generations, average_fitness[i], linewidth=2.0,
                color=colors[i], label='Population%d' % (i + 1))
    plt.legend(loc='upper left')
    plt.show()
    plt.savefig(path + '/fitness.png')


def plot_average_vs_best_fitness(average_fitness):
    """
    Plot the average fitness over all populations for all evolved
    generations and the best fitness of all populations.

    @param average_fitness: Average fitness per generation for each
                            population.
    """

    global path
    generations = range(1, 31)
    best = get_best_population(average_fitness)
    average = []
    for i in range(len(average_fitness[0])):
        average.append(np.mean([j[i] for j in average_fitness]))

    plt.figure()
    plt.subplot(1, 1, 1)
    plt.xlabel('Generations')
    plt.ylabel('Average Fitness Value')

    plt.plot(generations, average, linewidth=1.0, label='Average Fitness')
    plt.plot(generations, best, linewidth=2.0, label='Best Fitness')
    plt.legend(loc='upper left')
    plt.show()
    plt.savefig(path + '/average_fitness.png')
    plt.close()


def draw_network(topology):
    """
    Illustrate in a plot the presence/absence of connections between
    receptors and neurons and among neurons and whether the synapses
    are excitatory or inhibitory

    @param topology: Binary genetic string encoding the network
                        topology.
    """

    global path
    plt.figure('Network Topology')
    plt.subplot(1, 1, 1)
    plt.title('Synaptic Connections')
    plt.xticks(range(1, len(topology[0])))
    plt.yticks(range(1, len(topology) + 1))
    plt.xlabel('Neurons                                                       \
                Sensory Receptors')
    plt.ylabel('Neurons')

    for i in range(len(topology)):
        if topology[i][0]:
            synapse_color = 'Blue'
        else:
            synapse_color = 'Red'
        plt.scatter(0, i+1, s=32, color=synapse_color)

        for j in range(1, len(topology[0])):
            if topology[i][j]:
                plt.scatter(j, i+1, s=16, color='Black', marker='s')
        plt.axvline(x=0.5, ymax=10, color='black')
        plt.axvline(x=10.5, ymax=10, color='black')
    plt.show()


def plot_motor_neurons_membrane_potentials(data):
    """
    Plot the membrane potentials of the four motor neurons during the
    simulation time.

    @param: data: Voltmeter data
    """

    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, sharey=True)
    ax1.set_ylim(0, 0.12)
    ax1.set_ylabel('Membrane Potential')
    ax2.set_ylabel('Membrane Potential')
    ax3.set_ylabel('Membrane Potential')
    ax4.set_ylabel('Membrane Potential')
    ax4.set_xlabel('Time [ms]')

    ax1.plot(data[0][1], data[0][0])
    ax1.set_title('Left Forward Neuron')

    ax2.plot(data[1][1], data[1][0])
    ax2.set_title('Left Backward Neuron')

    ax3.plot(data[2][1], data[2][0])
    ax3.set_title('Right Forward Neuron')

    ax4.plot(data[3][1], data[3][0])
    ax4.set_title('Right Backward Neuron')

    f.subplots_adjust(hspace=0.3)


def plot_wheel_speeds(speed_log):
    """
    Plot the wheel speeds over time and save the speed_log array.

    @param speed_log: tuple comprising left wheel and right wheel
                        speeds' recorded during simulation.
    """

    t = range(len(speed_log))
    left_wheel = [j[0] for j in speed_log]
    right_wheel = [j[1] for j in speed_log]
    plt.figure()
    plt.subplot(1, 1, 1)
    plt.ylim(-40, 40)
    plt.plot(t, left_wheel, 'r', label='Left Wheel')
    plt.plot(t, right_wheel, 'b', label='Right Wheel')
    plt.legend(loc='lower right')
    plt.xlabel('Time [100 ms]')
    plt.ylabel('Speed [mm/s]')
    plt.title('Wheel Speeds')


def plot_motor_neurons_firing_rates(motor_fr_log):
    """
    Plot andsave the motor neurons firing rates log in a txt file.

    @param motor_fr_log: motor neurons firing rates log
    """

    t = range(len(motor_fr_log))
    left_fwd_fr = [j[0]*1000 for j in motor_fr_log]
    left_bwd_fr = [j[1]*1000 for j in motor_fr_log]
    right_fwd_fr = [j[2]*1000 for j in motor_fr_log]
    right_bwd_fr = [j[3]*1000 for j in motor_fr_log]

    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, sharey=True)
    ax1.set_ylabel('Rate [Hz]')
    ax2.set_ylabel('Rate [Hz]')
    ax3.set_ylabel('Rate [Hz]')
    ax4.set_ylabel('Rate [Hz]')
    ax4.set_xlabel('Time [100 ms]')
    ax1.set_ylim(0, 1000)

    ax1.plot(t, left_fwd_fr, linewidth=2.0)
    ax1.set_title('Left Forward Neuron')

    ax2.plot(t, left_bwd_fr, linewidth=2.0)
    ax2.set_title('Left Backward Neuron')

    ax3.plot(t, right_fwd_fr, linewidth=2.0)
    ax3.set_title('Right Forward Neuron')

    ax4.plot(t, right_bwd_fr, linewidth=2.0)
    ax4.set_title('Right Backward Neuron')

    f.subplots_adjust(hspace=0.3)


def plot_receptor_spiking_activity(receptor_sd):
    """
    Plot the receptors' spiking activity of the last simulated
    individual.
    """

    nest.raster_plot.from_device(receptor_sd,
                                 hist_binwidth=100.0, hist=False,
                                 title="Receptors' Spiking Activity")
    plt.ylim(10, 29)
    plt.yticks(range(11, 29))
    plt.ylabel("Receptor ID")


def plot_neurons_spiking_activity(neuron_sd):
    """
    Plot the spiking activity of the 10 neurons throughout the 40s.
    """

    nest.raster_plot.from_device(neuron_sd,
                                 hist_binwidth=100.0, hist=False,
                                 title="Neurons' Spiking Activity")
    plt.ylim(0, 11)
    plt.yticks(range(1, 11))


def get_best_population(average_fitness):
    """
    Determine the best population based on the average fitness over
    generations.

    @param average_fitness: Average fitness per generation for each
                            population.
    """

    average = [np.mean(j) for j in average_fitness]
    best_index = np.argmax(average)
    best = average_fitness[best_index]

    return best
    

def plot_fitness_function():
    """
    Plot the output of the fitness function for all possible left-wheel,
    right-wheel combination in a 3d plot.
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-40, 40)
    ax.set_xlabel('Left Wheel Speed [mm/s]')
    ax.set_ylim(-40, 40)
    ax.set_ylabel('Right Wheel Speed [mm/s]')

    x = range(-40, 41)
    y = range(-40, 41)
    x, y = np.meshgrid(x, y)

    z = np.zeros_like(x)
    for i in range(len(z)):
        for j in range(len(z[0])):
            z[i][j] = (x[i][j] + y[i][j]) if x[i][j] > 0 and y[i][j] > 0 \
            else 0

    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.jet,
                           linewidth=0)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def plot_avergae_spikes_per_second(spikes):
    """
    Plot the average spikes per second for each neuron during the
    simulation in a histogram.

    @param spikes: List containing the average number of spikes for each
                    neuron.
    """

    plt.figure()
    neurons = range(1, 11)
    x = [i - 0.4 for i in neurons]
    plt.bar(x, spikes)
    plt.xlim(0, 11)
    plt.xticks(neurons)
    plt.show()


def plot_eligibility_trace(tags, rec, nrn, color=None):
    """
    """

    label = "Receptor %d - Neuron %d" % (rec+11, nrn+1)
    tag = [tags[j][rec][nrn] for j in range(len(tags))]
    plt.plot(range(0, len(tags)*10, 10), tag, linewidth=2.0, color=color,
             label=label)


def plot_reward(reward):
    """
    """

    plt.figure()
    plt.plot(range(len(reward)), reward)
    plt.xlabel('Time [ms]')
    plt.ylabel('Reward')


def get_connections(src=None, tgt=None, mdl=None,
                    properties=('source', 'target', 'weight')):
    """
    Return synaptic connections fitlered according to given prameters.

    @param src: List of source GIDs
    @param tgt: List of target GIDs
    @param mdl: Synapse Model
    @param properties: Columns to extract from data.
    """

    conns = nest.GetConnections(source=src, target=tgt, synapse_model=mdl)
    conn_data = pd.DataFrame.from_records(
       list(nest.GetStatus(conns, keys=properties)), columns=properties)
    return conn_data


def get_num_exc_connections():
    """
    Calculate the number of excitatory connections in a network.
    """

    return nest.GetDefaults('e')['num_connections']


def get_num_inh_connections():
    """
    Calculate the number of inhibitory connections in a network.
    """

    return nest.GetDefaults('i')['num_connections']


def save_individual_results(fitness, traj, speed_log, linear, angular,
                            topology, motor_fr, p, g, i, x_init, y_init):              
    """
    Save both simulation results for each individual.
    """

    global path
    path_local = path + '/population%d' % (p + 1) + '/gen%d' % (g + 1) + \
                        '/individual%d' % (i + 1)

    # Determine if it's first or second run
    r = 2 if len(os.listdir(path_local)) > 0 else 1

    if r == 1:
        # Save network topology
        draw_network(topology)
        plt.savefig(path_local + '/topology.png')
    # Save wheel speeds' plot and txt
    plot_wheel_speeds(speed_log)
    plt.savefig(path_local + '/speeds%d.png' % r)
    np.savetxt(path_local + '/speed_log%d.txt' % r, speed_log)

    # Save trajectories
    plot_trajectory(traj, x_init, y_init)
    plt.savefig(path_local + '/traj%d.png' % r)
    
    # Save Motor Neurons firing rates plot and txt
    plot_motor_neurons_firing_rates(motor_fr)
    plt.savefig(path_local + '/motor_firing_rates%d.png' % r)

    # Save Receptor Spiking Activity
    plot_receptor_spiking_activity()
    plt.savefig(path_local + '/receptors%d.png' % r)

    # Save fitness value
    f = open(path_local+'/fitness%d.txt' % r, 'w+')
    f.write('fitness value: %f \n' % fitness)
    f.close()

    plt.close('all')


def save_generation_results(best, avg_fitness, avg_connect, gen):
    """
    Save plots and simulation data of the best individual of the
    generation.
    @param best: Best individual's simulation data.
    @param avg_fitness: Generation's average fitness.
    @param avg_connect: Generation's average connectivity.
    @param g: Generation number.
    """

    global path
    path_local = path + '/gen{}'.format(gen + 1)

    simData = best[3] if best[3]['fitness'] > best[4]['fitness'] else best[4]

    # Save speed_log plot and txt
    plot_wheel_speeds(simData['speed_log'])
    plt.savefig(path_local + '/speeds.png')
    np.savetxt(path_local + '/speed_log.txt', simData['speed_log'])
    
    # Save Network Topology
    draw_network(best[0])
    plt.savefig(path_local + '/topology.png')

    # Save trajecotry plot
    plot_trajectory(simData['traj'], simData['x_init'], simData['y_init'])
    plt.savefig(path_local + '/trajectory.png')

    # Save fitness value
    f = open(path_local+'/data.txt', 'w')
    f.write('Best Fitness Value: %f\n' % simData['fitness'])
    f.write('Avergae Fitness Value: %f\n' % avg_fitness[gen])
    f.write('Best Connectivity: %f\n' % best[2])
    f.write('Average Connectivity: %f\n' % avg_connect[gen])
    f.write('x_init: %d\ny_init: %d\ntheta_init: %f'\
    % (simData['x_init'], simData['y_init'], simData['theta_init']))
    f.close()

    # Save individual
    np.save(path_local + '/best_individual', best[0])
    np.save(path_local + '/simData', simData)

    plt.close('all')


def save_fitness(average_fitness, best_fitness):
    """
    Save simulation average fitness in a file.

    @param average_fitness: Average fitness per generation for each
                            population.
    """

    global path

    np.save(path+'/avg_fitness', average_fitness)
    np.save(path+'/best_fitness', best_fitness)


def save_rl_results(simdata, pop_spikes, i):
    """
    Save Results from reinforcement learning simulation.
    """

    path_local = path + '/run%d' % i
    os.mkdir(path_local)

#    plot_receptor_spiking_activity(pop_spikes[:1])
#    plt.savefig(path_local + '/receptors.png')
#
#    plot_neurons_spiking_activity(pop_spikes[1:])
#    plt.savefig(path_local + '/neurons.png')

    plot_trajectory(simdata['traj'], simdata['x_init'], simdata['y_init'])
    plt.savefig(path_local + '/trajectory.png')

#    plot_reward(simdata['reward'])
#    plt.savefig(path_local + '/reward.png')

    plot_wheel_speeds(simdata['speed_log'])
    plt.savefig(path_local + '/speeds.png')

    plt.close('all')


def plot_simulation_results(simData):
    """
    Plot data recorded from devices during simulation. This requires the
    NEST kernel not to be resetted after simulation ends.
    """

    # Plot motor neurons voltage trace.
    #plot_motor_neurons_membrane_potentials(simData['voltmeter_data'])

    # Plot Motor Neurons firing rates
    plot_motor_neurons_firing_rates(simData['motor_fr'])

    # Plot Receptor Spiking Activity
    plot_receptor_spiking_activity()

    # Plot Neurons' Spiking Activity
    plot_neurons_spiking_activity()

    # Plot Trajectory
    plot_trajectory(simData['traj'], simData['x_init'], simData['y_init'])
