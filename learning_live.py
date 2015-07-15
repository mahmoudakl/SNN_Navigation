import nest
import numpy as np
import learning
import network
import simulation as sim
import environment as env
import evolution as ev

n_step = 400
t_step = 100


if __name__ == '__main__':
    nest.SetKernelStatus({'resolution': 1.0})
    fitness, rewards, weights = [], [], []
    fitness.append(0)
    rewards.append(0)
    nrns, nrns_sd, rctrs, rctrs_sd, pop_spikes = learning.build_network()
    weights.append(learning.get_weights(rctrs, nrns))
    simtime = 0
    reward = 0
   
    data = []

    for run in range(10):
        print "RUN: %d" % run
        simdata = sim.create_empty_data_lists()
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
    
            nrns_st, rctrs_st = learning.get_spike_times(nrns_sd, rctrs_sd)
            rec_nrn_tags, nrn_nrn_tags, rctr_t, nrn_t, pt = learning.get_eligibility_trace(
                        nrns_st, rctrs_st, simtime,
                        simdata['rctr_nrn_trace'][t], simdata['nrn_nrn_trace'][t])
            
            simdata['rctr_nrn_trace'].append(rec_nrn_tags)
            simdata['nrn_nrn_trace'].append(nrn_nrn_tags)
            if (t+1) % 10 == 0:
                fitness = ev.get_fitness_value(simdata['speed_log'][t-10:])
                print "Fitness"
                print fitness
                print "Reward"
                reward = learning.get_reward(reward, fitness, x_cur, y_cur)
                print reward
                simdata['reward'].append(reward)
                delta_w_rec, delta_w_nrn = learning.get_weight_changes(reward, rec_nrn_tags,
                                                          nrn_nrn_tags)
                learning.update_weights(delta_w_rec, delta_w_nrn, nrns, rctrs)
                weights.append(learning.get_weights(rctrs, nrns))
            # Stop simulation if collision occurs
            if col:
    #            simdata['speed_log'].extend((0, 0) for k in range(t, 400))
    #            break
                print "COLLISION"
                break
                
        simdata['fitness'] = ev.get_fitness_value(simdata['speed_log'])
        data.append(simdata)
    #    reward = get_reward(reward, simdata['speed_log'], x_cur, y_cur)
    #    simdata['reward'].append(reward)