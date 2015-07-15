# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 22:50:49 2015

@author: akl
"""

import nest
import numpy as np
import environment as env
import evolution as ev
import network
import results


x_init = 195
y_init = 165
theta_init = 1.981011

g = 0

t_step = 100.0
n_step = 400

# Neuronal model
model = 'mat'
#model = 'iaf'

# Initial Pose
#init = 'fixed'
init = 'random'


def update_wheel_speeds(x_cur, y_cur, theta_cur, motor_firing_rates):
    """
    Update the left and right wheel speeds and calculate the linear and
    angular speeds of the robot.

    @param x_cur: Robot's current x position.
    @param y_cur: Robot's current y position.
    @param theta_cur: Robot's current orientation.
    """

    v_l, v_r = network.get_wheel_speeds(motor_firing_rates)
    r1, r2 = np.random.uniform(0.7, 1, 2)
    # get actual speeds by multiplyin the desired speeds with uniform
    # random variables
    v_l_act = v_l*r1
    v_r_act = v_r*r2
    v_l_act, v_r_act, collision = env.set_wheel_speeds(v_l_act, v_r_act, x_cur,
                                                       y_cur, theta_cur,
                                                       t_step)
    v_t = env.get_linear_velocity(v_l_act, v_r_act)
    w_t = env.get_angular_velocity(v_l_act, v_r_act, v_t)

    return v_l_act, v_r_act, v_t, w_t, 1 - r1, 1 - r2, collision, v_l, v_r


def create_empty_data_lists():
    """
    Create dict of empty lists to store simulation data in."""

    data = [[] for _ in range(12)]

    return {'traj':data[0], 'pixel_values': data[1], 'speed_log': data[2],
            'desired_speed_log': data[3], 'motor_fr': data[4],
            'linear_velocity_log': data[5], 'angular_velocity_log': data[6],
            'voltmeter_data': data[7], 'rctr_nrn_trace': data[8],
            'nrn_nrn_trace': data[9], 'reward': data[10], 'weights': data[11]}


def simulate(individual, reset=True):
    """
    Run simulation for 40 seconds based on network topology encoded in
    individual.

    @param individual: Binary string encoding network topology.
    @param reset: bool triggering the reset of the nest kernel.
    """

    global x_init, y_init, theta_init

    # Reset nest kernel just in case it was not reset & adjust
    # resolution.
    nest.ResetKernel()
    nest.SetKernelStatus({'resolution': 1.0, 'local_num_threads': 4,
                          'total_num_virtual_procs': 4})
    nest.set_verbosity('M_ERROR')

    if init == 'random':
        x_init = np.random.randint(1, env.x_max - 1)
        y_init = np.random.randint(1, env.y_max - 1)
        theta_init = np.pi*np.random.rand()

    simdata = create_empty_data_lists()
    simdata['x_init'] = x_init
    simdata['y_init'] = y_init
    simdata['theta_init'] = theta_init

    motor_spks = network.create_evolutionary_network(individual, model)
    x_cur = x_init
    y_cur = y_init
    theta_cur = theta_init
    err_l, err_r = 0, 0

    for t in range(n_step):
        # Save current position in trajectory list
        simdata['traj'].append((x_cur, y_cur))

        # Add nioise to
        network.update_refratory_perioud(model)

        # Set receptors' firing probability
        px = network.set_receptors_firing_rate(x_cur, y_cur, theta_cur,
                                               err_l, err_r)
        simdata['pixel_values'].append(px)

        # Run simulation for 100 ms
        nest.Simulate(t_step)

        motor_firing_rates = network.get_motor_neurons_firing_rates(motor_spks)

        # Get desired and actual speeds
        v_l_act, v_r_act, v_t, w_t, err_l, err_r, col, v_l_des, v_r_des = \
            update_wheel_speeds(x_cur, y_cur, theta_cur, motor_firing_rates)

        # Stop simulation if collision occurs
        if col:
            simdata['speed_log'].extend((0, 0) for k in range(t, 400))
            break

        # Save simulation data
        simdata['speed_log'].append((v_l_act, v_r_act))
        simdata['desired_speed_log'].append((v_l_des, v_r_des))
        simdata['motor_fr'].append(motor_firing_rates)
        simdata['linear_velocity_log'].append(v_t)
        simdata['angular_velocity_log'].append(w_t)

        # Move robot according to the read-out speeds from motor neurons
        x_cur, y_cur, theta_cur = env.move(x_cur, y_cur, theta_cur, v_t, w_t,
                                           t_step/1000.)

    # Calculate individual's fitness value
    simdata['fitness'] = ev.get_fitness_value(simdata['speed_log'])

    # Get average number of spikes per second for each neuron
    simdata['average_spikes'] = network.get_average_spikes_per_second()

    # Get Voltmeter data
    simdata['voltmeter_data'] = network.get_voltmeter_data()

    if reset:
        nest.ResetKernel()
    return simdata


def update_elite():
    """
    Keep track of the best performing individual throughout the whole
    simulation.
    """

    global elite, fitness, traj, individual, speed_log

    if fitness > elite[1]:
        elite[0] = individual
        elite[1] = fitness


def plot_results(average_fitness, elite):
    """
    Call all plotting functions.

    @param average_fitness: Average fitness per generation for each
                            population.
    @param elite: Tuple comprising best evolved individual's details.
    """

    results.plot_average_fitness_evolution(average_fitness)
    results.plot_average_vs_best_fitness(average_fitness)


if __name__ == '__main__':

    population, num_individuals = ev.load_population()
    generations = 30
    average_fitness, average_connectivity, best_of_generation = [], [], []
    elite = [0, 0]
    results.set_results_path(init, model, env.arena, ev.pop_id)
    print results.path

    for gen in range(generations):
        # Create a separate folder for each generation results
        results.create_generation_folder(gen)
        print 'generation: %d' % (gen + 1)
        generation_log = []
        for i in range(num_individuals):
            print i
            individual = population[i]
            simData1 = simulate(individual)
            simData2 = simulate(individual)
            fitness = np.mean([simData1['fitness'], simData2['fitness']])
            print 'fitness: %f %f %f' % (simData1['fitness'],
                                         simData2['fitness'], fitness)
            connectivity = ev.get_connectivity(population[i])
            generation_log.append((individual, fitness, connectivity, simData1,
                                   simData2, i))
        average_fitness.append(np.mean([j[1] for j in generation_log]))
        average_connectivity.append(np.mean([j[2] for j in  generation_log]))
        print 'Average Fitness: %f\n' % average_fitness[gen]
        top_performers = ev.get_top_performers(generation_log)
        best_of_generation.append(top_performers[0])
        update_elite()
        results.save_generation_results(top_performers[0], average_fitness,
                                        average_connectivity, gen)
        population = ev.evolve_new_generation(top_performers)
    results.save_fitness(average_fitness,
            [best_of_generation[i][1] for i in range(len(best_of_generation))])
    #plot_results(average_fitness, elite)
