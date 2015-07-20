# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 22:50:49 2015

@author: akl
"""

import nest

import numpy as np

import arena
import network
import results
import motion

import evolution as ev
import learning

x_init = 195
y_init = 165
theta_init = 1.981011

t_step = 100.0
n_step = 400


def create_empty_data_lists():
    """
    Create dict of empty lists to store simulation data in."""

    data = [[] for _ in range(12)]

    return {'traj':data[0], 'pixel_values': data[1], 'speed_log': data[2],
            'desired_speed_log': data[3], 'motor_fr': data[4],
            'linear_velocity_log': data[5], 'angular_velocity_log': data[6],
            'voltmeter_data': data[7], 'rctr_nrn_trace': data[8],
            'nrn_nrn_trace': data[9], 'reward': data[10], 'weights': data[11]}


def simulate_learning(neurons, receptors, nrns_sd, rctrs_sd, arena):
    """
    """

    global simtime
    simdata = create_empty_data_lists()

    if data['init'] == 'random':
        simdata['x_init'], simdata['y_init'], theta_init = \
                                                    select_random_pose(arena)
    x_cur = simdata['x_init']
    y_cur =  simdata['y_init']
    theta_cur = theta_init

    err_l, err_r = 0, 0

    simdata['rctr_nrn_trace'].append(np.zeros((18, 10)))
    simdata['nrn_nrn_trace'].append(np.zeros((10, 10)))
    motor_spks = nrns_sd[6:]

    for t in range(n_step):
        ev.update_refratory_perioud()
        simdata['traj'].append((x_cur, y_cur))

        px = network.set_receptors_firing_rate(x_cur, y_cur, theta_cur,
                                               err_l, err_r)
        simdata['pixel_values'].append(px)

        # Run simulation for time-step
        nest.Simulate(t_step)
        simtime += t_step
        motor_firing_rates = network.get_motor_neurons_firing_rates(motor_spks)

        v_l_act, v_r_act, v_t, w_t, err_l, err_r, col, v_l_des, v_r_des = \
        motion.update_wheel_speeds(x_cur, y_cur, theta_cur, motor_firing_rates)

        # Save dsired and actual speeds
        simdata['speed_log'].append((v_l_act, v_r_act))
        simdata['desired_speed_log'].append((v_l_des, v_r_des))

        # Save motor neurons firing rates
        simdata['motor_fr'].append(motor_firing_rates)

        # Save linear and angualr velocities
        simdata['linear_velocity_log'].append(v_t)
        simdata['angular_velocity_log'].append(w_t)

        # Move robot according to the read-out speeds from motor neurons
        x_cur, y_cur, theta_cur = motion.move(x_cur, y_cur, theta_cur, v_t,
                                              w_t)

        nrns_st, rctrs_st = learning.get_spike_times(nrns_sd, rctrs_sd)
        rec_nrn_tags, nrn_nrn_tags, rctr_t, nrn_t, pt = \
            learning.get_eligibility_trace(nrns_st, rctrs_st, simtime,
                    simdata['rctr_nrn_trace'][t], simdata['nrn_nrn_trace'][t])
        print simtime
        simdata['rctr_nrn_trace'].append(rec_nrn_tags)
        simdata['nrn_nrn_trace'].append(nrn_nrn_tags)

        # Stop simulation if collision occurs
        if col:
            simdata['speed_log'].extend((0, 0) for k in range(t, 400))
            break

    simdata['fitness'] = ev.get_fitness_value(simdata['speed_log'])

    return simdata, rctr_t, nrn_t, pt, col    


def simulate_evolution(individual, arena, reset=True):
    """
    Run simulation for 40 seconds based on network topology encoded in
    individual.

    @param individual: Binary string encoding network topology.
    @param reset: bool triggering the reset of the nest kernel.
    """

    # Reset nest kernel just in case it was not reset & adjust
    # resolution.
    nest.ResetKernel()
    nest.SetKernelStatus({'resolution': 1.0, 'local_num_threads': 4})
    nest.set_verbosity('M_ERROR')

    if data['init'] == 'random':
        x, y, theta = select_random_pose(arena)

    simdata = create_empty_data_lists()
    simdata['x_init'] = x_init
    simdata['y_init'] = y_init
    simdata['theta_init'] = theta_init

    motor_spks = network.create_evolutionary_network(individual, data['model'])
    x_cur = x_init
    y_cur = y_init
    theta_cur = theta_init
    err_l, err_r = 0, 0

    for t in range(n_step):
        # Save current position in trajectory list
        simdata['traj'].append((x_cur, y_cur))

        # Add nioise to
        network.update_refratory_perioud(data['model'])

        # Set receptors' firing probability
        px = network.set_receptors_firing_rate(x_cur, y_cur, theta_cur,
                                               err_l, err_r, arena)
        simdata['pixel_values'].append(px)

        # Run simulation for 100 ms
        nest.Simulate(t_step)

        motor_firing_rates = network.get_motor_neurons_firing_rates(motor_spks)

        # Get desired and actual speeds
        col, speed_dict = motion.update_wheel_speeds(x_cur, y_cur, theta_cur,
                                                 motor_firing_rates, t_step,
                                                 arena)

        # Stop simulation if collision occurs
        if col:
            simdata['speed_log'].extend((0, 0) for k in range(t, 400))
            break

        # Save simulation data
        simdata['speed_log'].append((speed_dict['actual_left'],
                                     speed_dict['actual_right']))
        simdata['desired_speed_log'].append((speed_dict['desired_left'],
                                             speed_dict['desired_right']))
        simdata['motor_fr'].append(motor_firing_rates)
        simdata['linear_velocity_log'].append(speed_dict['linear'])
        simdata['angular_velocity_log'].append(speed_dict['angular'])

        # Move robot according to the read-out speeds from motor neurons
        x_cur, y_cur, theta_cur = motion.move(x_cur, y_cur, theta_cur,
                    speed_dict['linear'], speed_dict['angular'], t_step/1000.)

    # Calculate individual's fitness value
    simdata['fitness'] = ev.get_fitness_value(simdata['speed_log'])

    # Get average number of spikes per second for each neuron
    simdata['average_spikes'] = network.get_average_spikes_per_second()

    # Get Voltmeter data
    simdata['voltmeter_data'] = network.get_voltmeter_data()

    if reset:
        nest.ResetKernel()
    return simdata


def select_random_pose(arena):
    """
    """

    x = np.random.randint(50, arena.maximum_length() - 50)
    y = np.random.randint(50, arena.maximum_width() - 50)
    theta = np.pi*np.random.rand()

    return x, y, theta


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


def get_simulation_details():
    """Get simulation specifications from the user."""

    data = {}
    mode = input(">>>Select Mode<<<\n[1] Evolution\n[2] Learning\n")
    mode = 'evolution' if mode == 1 else 'learning'
    data['mode'] = mode

    arena_no = input(">>>Select Arena<<<\n[1] Arena 1 (687x371)\n[2] Arena 2 \
(500x270)\n[3] Arena 3 (800x432)\n")
    data['arena'] = arena_no

    ar = arena.arena(arena_no)

    model = input(">>>Select Neuronal Model<<<\n[1] Multi-timescale Adaptive \
Threshold (MAT)\n[2] Leaky Intergate and Fire (LIF)\n")
    model = 'mat' if model == 1 else 'iaf'
    data['model'] = model

    init = input(">>>Select Input Pose<<<\n[1] Random\n[2] Fixed\n")
    init = 'random' if init == 1 else 'fixed'
    data['init'] = init

    if init == 'fixed':
        x = input(">>> Enter initial X-Position (max %d)<<<\n"""\
        % ar.maximum_length())
        y = input(">>>Enter initial Y-Position (max %d)<<<\n"""\
        % ar.maximum_width())
        theta = input(">>>Enter initial orientation angle<<<\n")
        data['x_init'] = x
        data['y_init'] = y
        data['theta_init'] = theta

    if mode == 'evolution':
        # Evolution
        pop_id = input(">>>Select Population id<<<\n[1] Population 1\n[2] \
Population 2\n[3] Population 3\n")
        data['population'] = pop_id
        
        generations = input(">>>Number of Generations<<<\n")
        data['generations'] = generations
    else:
        #Learning
        runs = input(">>>Number of Runs<<<\n")
        data['runs'] = runs

    return data


if __name__ == '__main__':

    data = get_simulation_details()
    arena = arena.arena(data['arena'])
    if data['mode'] == 'evolution':
        # Evolution
        population, num_individuals = ev.load_population(data['population'])
        average_fitness, average_connectivity, best_of_generation = [], [], []
        elite = [0, 0]
        results.set_results_path(data['init'], data['model'], data['arena'],
                                 data['population'])
        print results.path
        for gen in range(data['generations']):
            # Create a separate folder for each generation results
            results.create_generation_folder(gen)
            print 'generation: %d' % (gen + 1)
            generation_log = []
            for i in range(num_individuals):
                print i
                individual = population[i]
                simData1 = simulate_evolution(individual, arena)
                simData2 = simulate_evolution(individual, arena)
                fitness = np.mean([simData1['fitness'], simData2['fitness']])
                print 'fitness: %f %f %f' % (simData1['fitness'],
                                             simData2['fitness'], fitness)
                connectivity = ev.get_connectivity(population[i])
                generation_log.append((individual, fitness, connectivity,
                                       simData1, simData2, i))
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
    else:
        # Learning
        results.set_results_path(data['init'], data['model'], data['arena'],
                                 0)
        fitness, rewards, weights = [], [], []
        simtime = 0
        fitness.append(0)
        rewards.append(0)
        nrns, nrns_sd, rctrs, rctrs_sd, pop_spikes = \
                                network.create_learning_network(data['model'])
        for i in range(50):
            weights.append(learning.get_weights(rctrs, nrns))
            simdata, rt, nt, pt, col = simulate_learning(nrns, rctrs, nrns_sd,
                                                         rctrs_sd, arena)
            fitness.append(simdata['fitness'])
            reward = learning.get_reward(rewards[-1], fitness[-1],
                                simdata['traj'][-1][0], simdata['traj'][-1][1])
            rewards.append(reward)
            delta_w_rec, delta_w_nrn = learning.get_weight_changes(reward,
                simdata['rctr_nrn_trace'][-1], simdata['nrn_nrn_trace'][-1])
            learning.update_weights(delta_w_rec, delta_w_nrn, nrns, rctrs)
            results.save_rl_results(simdata, pop_spikes, i)
