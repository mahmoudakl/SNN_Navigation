# -*- coding: utf-8 -*-
"""
Created on Fri May 29 23:12:06 2015

@author: akl
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = (12, 4)

def build_network(Ie):
    """
    """

    nest.ResetKernel()
    n = nest.Create('iaf_neuron', params={'I_e': Ie})
    vm = nest.Create('voltmeter', params={'interval': 0.1})
    sd = nest.Create('spike_detector')

    nest.Connect(vm, n)
    nest.Connect(n, sd)

    return vm, sd


def build_network():
    """
    """

    model = 'mat'
    nest.ResetKernel()
    nest.SetKernelStatus({'resolution': .01, 'local_num_threads': 4})
    if model == 'mat':
        neuron_params = {'E_L': 0.0, 'V_m': 0.0, 'tau_m': 4.0,
                         'tau_syn_ex': 10.0, 'tau_syn_in': 3.0,
                         'alpha_1': 1.0, 'alpha_2': 0.0, 't_ref': 0.1,
                         'C_m': 50.0, 'omega': 0.1, 'tau_1': 4.0}
        n = nest.Create('mat2_psc_exp', params=neuron_params)
    else:
        neuron_params = {'V_m': 0.0, 'E_L': 0.0, 'C_m': 50.0, 'tau_m': 4.0,
                         't_ref': np.random.uniform(0.1, 2), 'V_th': 0.3,
                         'V_reset': np.random.uniform(0, -1), 'tau_syn': 10.0}
        n = nest.Create('iaf_neuron', params=neuron_params)

    vm = nest.Create('voltmeter', params={'interval': 0.01})
    sg1 = nest.Create('spike_generator')
    sg2 = nest.Create('spike_generator')

    nest.CopyModel('static_synapse_hom_w', 'e',
                   {'weight': 1.0, 'delay': 2.0})
    nest.CopyModel('static_synapse_hom_w', 'i',
                   {'weight': -500.0, 'delay': 1.0})
    nest.Connect(vm, n)
    nest.Connect(sg1, n, syn_spec='e')
    nest.Connect(sg2, n, syn_spec='i')

    return vm, sg1, sg2


def simulate_network():
    """
    """

    vm, sg1, sg2 = build_network()
    nest.SetStatus(sg1, {'spike_times': [0.1]})

    nest.Simulate(30)
    vme = nest.GetStatus(vm, 'events')[0]
    V, t = vme['V_m'], vme['times']
    plt.plot(t, V, label='tau_syn=10.0, C_m=50.0', linewidth=2.0)
