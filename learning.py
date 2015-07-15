# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 17:46:27 2015

@author: akl
"""

import numpy as np

import network
import environment as env
import simulation as sim
import evolution as ev
import results

import nest

init = 'fixed'
#init = 'random'

model = 'mat'
#model = 'iaf'

x_init = 195
y_init = 65
theta_init = 1.981011

x_cur = 0
y_cur = 0

t_step = 100.0
n_step = 400
tau = 10.
tau_c = 10.
tau_d = 100.
simtime = 0


def build_network():
    """
    Create NEST Network with 18 receptors, 10 neurons and spik
    detectors. Establish connections with random initial weights between
    all nodes.
    """

    if model == 'mat':
        neuron_params = {'E_L': 0.0, 'V_m': 0.0, 'tau_m': 4.0, 'C_m': 10.0,
                         'tau_syn_ex': 3.0, 'tau_syn_in': 3.0, 'omega': 0.1,
                         'alpha_1': 1.0, 'alpha_2': 0.0,
                         't_ref': 0.1, 'tau_1': 4.0}
        # 10 mat neurons
        neurons = nest.Create('mat2_psc_exp', 10, neuron_params)
    else:
        neuron_params = {'V_m': 0.0, 'E_L': 0.0, 'C_m': 50.0, 'tau_m': 4.0,
                         't_ref': np.random.uniform(0.1, 2), 'V_th': 0.1,
                         'V_reset': np.random.uniform(0, -1), 'tau_syn': 10.0}
        neurons = nest.Create('iaf_neuron', 10, neuron_params)

    # 18 poisson generators representing neural receptors
    receptors = nest.Create('spike_generator', 18)

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
                                             'low': 0.0, 'high': 1.5}})
    nest.Connect(neurons, neurons, {'rule': 'all_to_all'},
                 {'model': 'syn', 'weight': {'distribution': 'uniform',
                                             'low': -1.5, 'high': 1.5}})

    return neurons, neuron_spike_detectors, receptors,\
        receptor_spike_detectors, population_spikes


def build_small_network():
    """
    Create test network with one neuron and one receptor.
    """
    neuron_params = {'E_L': 0.0, 'V_m': 0.0, 'tau_m': 4.0, 'C_m': 10.0,
                         'tau_syn_ex': 3.0, 'tau_syn_in': 3.0, 'omega': 0.1,
                         'alpha_1': 1.0, 'alpha_2': 0.0,
                         't_ref': 0.1, 'tau_1': 4.0}
    neuron = nest.Create('mat2_psc_exp', 1, neuron_params)
    receptor = nest.Create('spike_generator', 1)

    n_sd = nest.Create('spike_detector', 1)
    r_sd = nest.Create('spike_detector', 1)

    nest.CopyModel('static_synapse', 'e', {'delay': 2.0, 'weight': 1.0})
    nest.CopyModel('static_synapse', 'syn', {'delay': 2.0})
    
    nest.Connect(neuron, n_sd)
    nest.Connect(receptor, r_sd, syn_spec={'model': 'syn'})
    nest.Connect(receptor, neuron, syn_spec={'model': 'syn'})

    return neuron, receptor, n_sd, r_sd


def sim_small_network():
    """Simulatethe test network."""

    simtime = 0
    rec_e_trace = np.zeros(n_step)
    nrn, rctr, n_sd, r_sd = build_small_network()
    nest.SetStatus(rctr, {'spike_times': [1.0, 11.0, 21.0, 31.0, 41.0, 51.0]})
    for i in range(7):
        prev_tag = 0 if i == 0 else rec_e_trace[i-1]
        nest.Simulate(n_step)
        simtime += 10
        n_st, r_st = get_spike_times(n_sd, r_sd)
        syn, rctr_t, nrn_t = get_eligibility_trace(n_st, r_st, simtime,
                                                    prev_tag)
        rec_e_trace[i] = syn
    return rec_e_trace, rctr_t, nrn_t


def simulate(neurons, receptors, nrns_sd, rctrs_sd):
    """
    """

    global simtime, x_cur, y_cur, x_init, y_init, theta_init
    simdata = sim.create_empty_data_lists()

    if init == 'random':
        x_init = np.random.randint(50, env.x_max - 50)
        y_init = np.random.randint(50, env.y_max - 50)
        theta_init = np.pi*np.random.rand()
    x_cur = x_init
    y_cur = y_init
    theta_cur = theta_init
    simdata['x_init'] = x_cur
    simdata['y_init'] = y_cur

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
        sim.update_wheel_speeds(x_cur, y_cur, theta_cur, motor_firing_rates)

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

        nrns_st, rctrs_st = get_spike_times(nrns_sd, rctrs_sd)
        rec_nrn_tags, nrn_nrn_tags, rctr_t, nrn_t, pt = get_eligibility_trace(
                    nrns_st, rctrs_st, simtime,
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


def get_reward(reward, fitness, x, y):
    """
    Get reward signal based on robot's wheel speeds and proximity to
    walls.

    @param reward: Previous reward signal.
    @param speed_log: Wheel speeds during simualtion.
    @param x: Current x-position.
    @param y: Current y-position.
    """

    reward = reward*np.exp(-t_step/tau_d)
    if fitness == 0:
        return reward - 1
    else:
        if env.proximity_to_wall(x, y) > 5:
            return reward + 1
        else:
            return reward - 1


def get_weights(receptors, neurons):
    """
    Get current synaptic weights of synapses connecting receptors to
    neurons and neurons to neurons.
    """

    conns = nest.GetConnections(source=receptors+neurons, target=neurons)
    weights = nest.GetStatus(conns, 'weight')

    return weights


def get_weight_changes(reward, rec_nrn_trace, nrn_nrn_trace):
    """
    Get the weight changes for all synapses based on the global reward
    signal and the eligibility trace of each synapse.

    @param reward: Global reward signal
    @param rec_nrn_trace: List of current eligibility trace values for
                            all synapses between receptors and neurons.
    @param nrn_nrn_trace: List of current eligibility trace values for
                            all synapses among neurons.
    """

    delta_w_rec_nrn = rec_nrn_trace*reward
    delta_w_nrn_nrn = nrn_nrn_trace*reward

    return delta_w_rec_nrn, delta_w_nrn_nrn


def update_weights(delta_w_rec, delta_w_nrn, neurons, receptors):
    """
    Update the synaptic weights for all connections.

    @param delta_w_rec: Weight changes for synapses connecting receptors
                        and neurons.
    @param delta_w_nrn: Weight chnages for synapses connecting neurons
                        to eachother.
    @param neurons: List of global identifiers of the 10 neurons.
    @param receptors: List of global identifiers of the 18 receptors.
    """

    for i in range(len(delta_w_rec)):
        for j in range(len(delta_w_rec[i])):
            source = [receptors[i]]
            target = [neurons[j]]
            conn = nest.GetConnections(source=source, target=target)
            weight = nest.GetStatus(conn, 'weight')[0] + delta_w_rec[i][j]
            weight = 3.0 if weight > 3.0 else weight
            weight = -3.0 if weight < -3.0 else weight
            nest.SetStatus(conn, {'weight': weight})

    for i in range(len(delta_w_nrn)):
        for j in range(len(delta_w_nrn[i])):
            source = [neurons[i]]
            target = [neurons[j]]
            conn = nest.GetConnections(source=source, target=target)
            weight = nest.GetStatus(conn, 'weight')[0] + delta_w_nrn[i][j]
            nest.SetStatus(conn, {'weight': weight})


def get_spike_times(nrns_sd, rctrs_sd):
    """
    Extract spike times for receptors and neurons from the spike
    detecotrs.
    """

    nrns_st, rctrs_st = [], []
    for i in nrns_sd:
        nrns_st.append(nest.GetStatus([i])[0]['events']['times'])
    for i in rctrs_sd:
        rctrs_st.append(nest.GetStatus([i])[0]['events']['times'])

    return nrns_st, rctrs_st


def stdp(t):
    """
    Calculate the STDP based on the time difference between pre- and
    postsynaptic neuron.

    @param t: Inter-spike interval. 
    """

    if t > 0:
        return np.exp(-t/tau) 
    elif t < 0:
        return -np.exp(t/tau)
    else:
        return 0


def get_syn_tag(c, delta_t):
    """
    Get the change in the eligibtility trace for a connection based on
    the inter-spike interval.

    @param c: current value of the egligibility trace.
    @param delta_t: inter-spike interval.
    """

    return c + stdp(delta_t)


def decay_syn_tag(syn_tag, t):
    """
    Get the value of the eligibility trace for a connection due to
    inactivity for some time.

    @param syn_tag: Last value for the eligibility trace.
    @param t: Duration of inactivity.
    """

    syn_tag = syn_tag*np.exp(-t/tau_c)
    return syn_tag


def get_eligibility_trace(nrns_st, rctrs_st, simtime, rec_nrn_tags,
                           nrn_nrn_tags):
    """
    Get the current value of the eligibility trace for all synapses in
    the network based on the spike trains.
    """

    rctr_times, nrn_times, nrn_prev_times = [], [], []
    tags180 = rec_nrn_tags.copy()
    tags100 = nrn_nrn_tags.copy()

    for i in range(len(rctrs_st)):
        rctr = rctrs_st[i]
        times = rctr[rctr > (simtime - t_step)]
        rctr_times.append(times)
    for i in range(len(nrns_st)):
        nrn = nrns_st[i]
        times = nrn[nrn > (simtime - t_step)]
        prev_times = nrn[nrn > (simtime - 2*t_step)]
        prev_times = prev_times[prev_times < (simtime - t_step)]
        nrn_times.append(times)
        nrn_prev_times.append(prev_times)

    # Correlations between Receptors and Neurons
    for i in range(len(rctr_times)):
        if len(rctr_times[i]) == 0:
            for j in range(10):
                tags180[i][j] = decay_syn_tag(tags180[i][j], int(t_step))
            continue
        for j in range(len(nrn_times)):
            if len(nrn_prev_times[j]) != 0:
                time = nrn_prev_times[j][-1] - rctr_times[i]
                tags180[i][j] = get_syn_tag(tags180[i][j], time)
            if len(nrn_times[j]) == 0:
                tags180[i][j] = decay_syn_tag(tags180[i][j], int(t_step))
                continue
            for t in range(len(nrn_times[j])):
                spike_time = nrn_times[j][t]
                delta_t = spike_time - rctr_times[i][0]                
                if t == 0:
                    decay_time = spike_time - (simtime - t_step)
                else:
                    prev_spike_time = nrn_times[j][t-1]
                    decay_time = spike_time - prev_spike_time
                tags180[i][j] = decay_syn_tag(tags180[i][j], int(decay_time))
                tags180[i][j] = get_syn_tag(tags180[i][j], delta_t)
            decay_time = simtime - spike_time
            tags180[i][j] = decay_syn_tag(tags180[i][j], int(decay_time))

    # Correlations among neurons
    for i in range(len(nrn_times)):
        for j in range(len(nrn_times)):
            times = get_both_nodes_times(nrn_times[i], nrn_times[j])
            times_prev = get_both_nodes_times(nrn_prev_times[i],
                                              nrn_prev_times[j])
            if not times:
                tags100[i][j] = decay_syn_tag(tags100[i][j], int(t_step))
            else:
                interspike = get_interspike_intervals(times, times_prev)
                for t in range(len(interspike)):
                    if t == 0:
                        decay_time = times[t][1] - (simtime - t_step)
                    else:
                        decay_time = interspike[t][0] - interspike[t - 1][0]
                    tags100[i][j] = decay_syn_tag(tags100[i][j],
                                                    int(decay_time))
                    tags100[i][j] = get_syn_tag(tags100[i][j], interspike[t][1])
            if len(times) > 0:
                decay_time = simtime - times[-1][1]
            else:
                decay_time = t_step
            tags100[i][j] = decay_syn_tag(tags100[i][j], int(decay_time))

    return tags180, tags100, rctr_times, nrn_times, nrn_prev_times


def get_both_nodes_times(pre, post):
    """
    Order the spike times of two connected neurons in one list,
    specifying whether the neurons is the pre- or the postsynaptic one.

    @param pre: Spike times of the presynaptic neuron.
    @param post: Spike times of the postsynaptic neuron.
    """

    result = []
    count_pre = 0
    count_post = 0
    for i in range(len(pre) + len(post)):
        if count_pre == len(pre):
            result.append(('post', post[count_post]))
            count_post += 1
        elif count_post == len(post):
            result.append(('pre', pre[count_pre]))
            count_pre += 1
        elif pre[count_pre] <= post[count_post]:
            result.append(('pre', pre[count_pre]))
            count_pre += 1
        else:
            result.append(('post', post[count_post]))
            count_post += 1

    return result


def get_interspike_intervals(spike_times, prev_times):
    """
    Get the interspike itervals between two spike trains of a pre- and
    a postsynaptic neuron. Positive intervals indicate that the interval
    is between a pre- and a postsynaptic spike. Negative intervals
    indicate the other way round.

    @param spike_times: Spike times of the current time-step.
    @param prev_times: Spike times of the previous time-step.
    """

    intervals = []
    times = prev_times + spike_times
    for i in range(len(spike_times)):
        interval = ()
        if spike_times:
            if spike_times[i][0] == 'post':
                prev = get_previous_spike(times, spike_times[i][1], 'pre')
                if prev != -1:
                    dt = spike_times[i][1] - prev
                    interval = (spike_times[i][1], dt)
            else:
                prev = get_previous_spike(times, spike_times[i][1], 'post')
                if prev != -1:
                    dt = prev - spike_times[i][1]
                    interval = (spike_times[i][1], dt)
            if interval:
                intervals.append(interval)

    return intervals


def get_previous_spike(spike_times, current_time, spike_type):
    """
    Get the previous pre- or post synaptic spike to a give spike time.

    @param spike_times: Spike times of the current time-step and the
                        time-step before.
    @param current_time: Time of the spike time, for which the previous
                            spike is requested.
    @param spike_type: Indication of whether the requested previous
                        spike is a pre- or a postsynaptic one.
    """

    spike_times = spike_times[::-1]
    for i in range(len(spike_times)):
        if spike_times[i][1] == current_time:
            for j in range(i + 1, len(spike_times)):
                if spike_times[j][0] == spike_type:
                    if spike_times[j][1] != current_time:
                        return spike_times[j][1]
                    else:
                        continue
            return -1


if __name__ == '__main__':
    results.set_results_path(init, model, env.arena, 0)
    nest.SetKernelStatus({'resolution': 1.0})
    fitness, rewards, weights = [], [], []
    fitness.append(0)
    rewards.append(0)
    nrns, nrns_sd, rctrs, rctrs_sd, pop_spikes = build_network()

    for i in range(50):
        weights.append(get_weights(rctrs, nrns))
        simdata, rt, nt, pt, col = simulate(nrns, rctrs, nrns_sd, rctrs_sd)
        fitness.append(simdata['fitness'])
        reward = get_reward(rewards[-1], fitness[-1], x_cur, y_cur)
        rewards.append(reward)
        delta_w_rec, delta_w_nrn = get_weight_changes(reward, simdata['rctr_nrn_trace'][-1],
                                                      simdata['nrn_nrn_trace'][-1])
        update_weights(delta_w_rec, delta_w_nrn, nrns, rctrs)
        results.save_rl_results(simdata, pop_spikes, i)
