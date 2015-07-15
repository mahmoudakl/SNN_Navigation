# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 15:53:40 2015

@author: akl
"""
import learning
import simulation as sim
import environment as env
import network
import results

import nest

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x_init = 195
y_init = 65
theta_init = 1.981011

def create_empty_data_lists():
    """
    Create dict of empty lists to store simulation data in.
    """

    data = [[] for _ in range(8)]

    return {'traj':data[0], 'pixel_values': data[1], 'speed_log': data[2],
            'desired_speed_log': data[3], 'motor_fr': data[4],
            'linear_velocity_log': data[5], 'angular_velocity_log': data[6],
            'voltmeter_data': data[7]}


def get_motor_neurons_firing_rates(motor_spikes):
    """
    """

    left_fwd = motor_spikes[:1]
    left_bwd = motor_spikes[1:2]
    right_fwd = motor_spikes[2:3]
    right_bwd = motor_spikes[3:]

    # Spike times for the left and right wheel neurons
    left_fwd_events = nest.GetStatus(left_fwd)[0]['events']['times']
    left_bwd_events = nest.GetStatus(left_bwd)[0]['events']['times']
    right_fwd_events = nest.GetStatus(right_fwd)[0]['events']['times']
    right_bwd_events = nest.GetStatus(right_bwd)[0]['events']['times']

    left_fwd_fr = network.get_neuron_firing_rate(left_fwd_events)
    left_bwd_fr = network.get_neuron_firing_rate(left_bwd_events)
    right_fwd_fr = network.get_neuron_firing_rate(right_fwd_events)
    right_bwd_fr = network.get_neuron_firing_rate(right_bwd_events)

    firing_rates = (left_fwd_fr, left_bwd_fr, right_fwd_fr, right_bwd_fr)

    return firing_rates


def simulate(nrns, recs, pop_spikes, motor_spikes, t_step, n_step):
    """
    """

    ini_wgts_rec = nest.GetStatus(nest.GetConnections(source=recs,
                                                      synapse_model='plastic'),
                                                      'weight')
    ini_wgts_nrn = nest.GetStatus(nest.GetConnections(source=nrns,
                                                      synapse_model='plastic'),
                                                      'weight')
    e_wgts_rec = np.nan*np.ones((len(ini_wgts_rec), n_step+1))
    e_wgts_rec[:, 0] = ini_wgts_rec
    e_wgts_nrn = np.nan*np.ones((len(ini_wgts_nrn), n_step+1))
    e_wgts_nrn[:, 0] = ini_wgts_nrn

    t_wgts = t_step * np.arange(n_step+1)

    simdata = create_empty_data_lists()
    simdata['x_init'] = x_init
    simdata['y_init'] = y_init
    simdata['theta_init'] = theta_init

    x_cur = x_init
    y_cur = y_init
    theta_cur = theta_init
    err_l, err_r = 0, 0

    for t in range(n_step):
        simdata['traj'].append((x_cur, y_cur))
        px = network.set_receptors_firing_rate(x_cur, y_cur, theta_cur,
                                               err_l, err_r)
        simdata['pixel_values'].append(px)

        nest.Simulate(t_step)

        motor_firing_rates = get_motor_neurons_firing_rates(motor_spikes)

        v_l_act, v_r_act, v_t, w_t, err_l, err_r, col, v_l_des, v_r_des = \
        sim.update_wheel_speeds(x_cur, y_cur, theta_cur, motor_firing_rates)

        if col:
            simdata['speed_log'].extend((0, 0) for k in range(t, 400))
            break

        reward = learning.get_reward(v_l_des, v_r_des)
        print reward
        #learning.update_learning_rate(reward)

        # Save dsired and actual speeds
        simdata['speed_log'].append((v_l_act, v_r_act))
        simdata['desired_speed_log'].append((v_l_des, v_r_des))

        # Save motor neurons firing rates
        simdata['motor_fr'].append(motor_firing_rates)

        # Save linear and angualr velocities
        simdata['linear_velocity_log'].append(v_t)
        simdata['angular_velocity_log'].append(w_t)

        # Move robot according to the read-out speeds from motor neurons
        x_cur, y_cur, theta_cur = env.move(x_cur, y_cur, theta_cur, v_t, w_t)

        rec_spks = nest.GetStatus(pop_spikes[:1], 'events')[0]
        nrn_spks = nest.GetStatus(pop_spikes[1:], 'events')[0]

        e_wgts_rec[:, t+1] = nest.GetStatus(nest.GetConnections(source=recs,
                                                      synapse_model='plastic'),
                                                      'weight')
        
        e_wgts_nrn[:, t+1] = nest.GetStatus(nest.GetConnections(source=nrns,
                                                      synapse_model='plastic'),
                                                      'weight')

        nrn_spks = pd.DataFrame(nrn_spks)
        rec_spks = pd.DataFrame(rec_spks)

    return simdata, e_wgts_rec, e_wgts_nrn, t_wgts, nrn_spks, rec_spks


def get_results(nrn_spks, rec_spks, t_wgts, e_wgts, nrec=10):
    
    dt = t_wgts[1] - t_wgts[0]

    num_spikes = lambda sp, t: np.array([len(sp[(t[k] < nrn_spks.times) & (nrn_spks.times <= t[k+1])]) 
                                         for k in range(len(t)-1)])
    
    rates = pd.DataFrame(
               {'Time [ms]': t_wgts[1:],
                'Excitatory rate [Hz]': num_spikes(nrn_spks, t_wgts)  / (dt/1000. * nrec),
                'Inhibitory rate [Hz]': num_spikes(nrn_spks, t_wgts)  / (dt/1000. * nrec)})

    print rates
        
    plt.subplot(311)
    plt.hist(e_wgts, histtype='step', bins=100)
    
    plt.subplot(312)
    e_gid_min = nrn_spks.senders.min()
    i_gid_min = rec_spks.senders.min()
    e_plot = (nrn_spks.senders < (e_gid_min + 40))
    i_plot = (rec_spks.senders < (i_gid_min + 10))
    plt.plot(nrn_spks.times[e_plot], nrn_spks.senders[e_plot] - e_gid_min +  1, 'bo', markersize=2, markeredgecolor='none')
    plt.plot(rec_spks.times[i_plot], rec_spks.senders[i_plot] - i_gid_min + 41, 'ro', markersize=2, markeredgecolor='none')
    plt.ylim(0, 51)
    plt.yticks([])
    
    plt.subplot(313)
    bins = np.arange(0, 2000, 5.)
    plt.hist(nrn_spks.times.values, bins=bins, histtype='step', color='b', lw=2, label='Exc')
    plt.hist(rec_spks.times.values, bins=bins, histtype='step', color='r', lw=2, label='Inh')
    plt.legend();


if __name__ == '__main__':

    nest.SetKernelStatus({'resolution': 1.0, 'local_num_threads': 4})
    t_step = 1.
    n_step = 200
    spike_detectors, nrns, recs, motor_spikes = learning.create_network()
    simdata, w_rec, w_nrn, t_wgts, nrn_spks, rec_spks = \
        simulate(nrns, recs, spike_detectors, motor_spikes, t_step, n_step)
    t_wgts = t_step*np.arange(n_step + 1)
