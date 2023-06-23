import nest
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import nestShow as ns
from nestNeuralNet import nestNet
from decimal import Decimal

textSize = 18
plt.rcParams.update({'font.size': 12,
                     "text.usetex": True})


def run(params):
    nest.set_verbosity(18)  # nie pokazuj info od nest

    net = type('', (), {})()

    config = {'neuronModel': "iaf_psc_alpha",
              'generatorModel': "dc"}
    nestNet.config_nest(params)
    nestNet.refresh_nest()

    param = {'I_e': 1000.0}
    net.Vneurons = nest.Create(config['neuronModel'], params=param)

    net.Wneurons = nest.Create(config['neuronModel'])

    conn_spec_dict = {"rule": "all_to_all"}
    syn_spec_dict = {"weight": 10.0,
                     "model": "static_synapse",
                     "delay": nestNet.dt,
                     }
    nest.Connect(net.Vneurons, net.Wneurons, conn_spec=conn_spec_dict, syn_spec=syn_spec_dict)

    # dodaj oberwatory
    what2rec = ['I_syn_in', 'I_syn_ex', 'V_m']  # 'weighted_spikes_in', 'weighted_spikes_ex']
    multimetersParams = {"withtime": True,
                         "record_from": what2rec,
                         'interval': nestNet.dt}
    spikedetectorParams = {"withtime": True}

    net.Vmultimeters = nest.Create('multimeter', 1, params=multimetersParams)
    nest.Connect(net.Vmultimeters, net.Vneurons, "one_to_one", syn_spec=syn_spec_dict)
    net.Vspikedetectors = nest.Create('spike_detector', 1, params=spikedetectorParams)
    nest.Connect(net.Vneurons, net.Vspikedetectors, "one_to_one", syn_spec=syn_spec_dict)

    # wyjsciowych
    net.Wmultimeters = nest.Create('multimeter', 1, params=multimetersParams)
    nest.Connect(net.Wmultimeters, net.Wneurons, "one_to_one", syn_spec=syn_spec_dict)
    net.Wspikedetectors = nest.Create('spike_detector', 1, params=spikedetectorParams)
    nest.Connect(net.Wneurons, net.Wspikedetectors, "one_to_one", syn_spec=syn_spec_dict)

    nest.Simulate(20 + 0.01)
    whatShow = ["multimeter.I_syn_ex", "multimeter.I_syn_in", "multimeter.V_m",
                "spikedetector.senders"]
    return nest.GetStatus(net.Wmultimeters)[0]['events']['I_syn_ex']
    # a = nest.GetStatus(net.Vmultimeters)
    # ns.showNeurons(net.Vneurons+net.Wneurons, net, whatShow)


r = {}
all_tau = [3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0]
for tau in all_tau:
    params = {'tau_syn_ex': tau,
              'tau_syn_in': tau}
    r[tau] = run(params)

t = np.arange(0, 20, 0.01)
plt.figure(1)
i = 0
for tau, I in r.items():
    l = r'$\tau_s = '+str(tau)+r'$'
    plt.plot(t, I, label=l,  linewidth=2.5)
plt.xlabel(r't[ms]')
plt.ylabel(r'I[pA]')
plt.legend()
plt.show()

# dt = nest.GetStatus(net.Vspikedetectors)[0]['events']['times'][0]
# target = nest.GetStatus(net.Wmultimeters)[0]['events']['V_m']
# shift = [0] * (round(dt/nestNet.dt))
# odp = nest.GetStatus(net.Vmultimeters)[0]['events']['V_m']
# # odp = shift + nestNet.PSC_ex.tolist()
# # odp = odp[:1999]
# plt.plot(target, label="target")
# plt.plot(odp, label="calc")
# plt.legend()
# plt.show()
