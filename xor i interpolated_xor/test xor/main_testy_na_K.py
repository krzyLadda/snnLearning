import nest
import matplotlib.pyplot as plt
import numpy as np
from nestNeuralNet import nestNet
import math
import pickleData
from parallel import parallel_for_online_learning
import os


def run(task_id, params=None):
    np.random.seed()
    nest.set_verbosity(18)  # nie pokazuj info od nest
    nestNet.config_nest(params)

    load_net = None
    if load:
        print('load')

    elif what_to_teach == "xor":
        print('xor')
        hmIn = 3
        hmOut = 1
        hmN = 5

        version = 'online'
        hm_workers = 4

        hmEpochs = 500
        target_MSE_spikes = 0.25
        target_learning_MSE = None
        target_test_MSE = None

        learning_rate = 0.5

        learning_inputs = [[1, 1, 1],  # 1 1
                           [0, 1, 1],  # 0 1
                           [1, 0, 1],  # 1 0
                           [0, 0, 1]]  # 0 0
        learning_inputs = np.array(learning_inputs)

        learning_targets = [[0],  # 0
                            [1],  # 1
                            [1],  # 1
                            [0]]  # 0
        learning_targets = np.array(learning_targets)

        test_inputs = None
        test_targets = None

        config = {}

    # do zapisu
    config['what_to_teach'] = what_to_teach
    config['learning_inputs'] = learning_inputs
    config['learning_targets'] = learning_targets
    config['test_inputs'] = test_inputs
    config['test_targets'] = test_targets
    config['hmIn'] = hmIn
    config['hmOut'] = hmOut
    config['hmN'] = hmN
    config['learning_rate'] = learning_rate
    config['hmEpochs'] = hmEpochs
    config['target_learning_MSE'] = target_learning_MSE
    config['target_test_MSE'] = target_test_MSE
    config['target_MSE_spikes'] = target_MSE_spikes
    config['version'] = version

    if version == 'batch' and use_parallel:
        raise 'Do not use two parralels in one time.'
    # stworz lub wycztaj sieć
    if load_net is not None:
        net = load_net
    else:
        learning_MSE_gen = []
        test_MSE_gen = []
        learning_MSE_spikes_gen = []
        net = nestNet(hmIn, hmOut, hmN=hmN)

    net.train(learning_inputs, learning_targets, learning_rate, hmEpochs=hmEpochs,
              target_learning_MSE=target_learning_MSE,
              target_test_MSE=target_test_MSE,
              test_inputs=test_inputs, test_targets=test_targets,
              learning_MSE_gen=learning_MSE_gen, test_MSE_gen=test_MSE_gen,
              task_id=task_id, dynamic=False, version=version, hm_workers=hm_workers,
              learning_MSE_spikes_gen=learning_MSE_spikes_gen, target_MSE_spikes=target_MSE_spikes)

    save_k = [net, learning_inputs, learning_targets, test_inputs, test_targets, config,
              learning_MSE_gen, test_MSE_gen, learning_MSE_spikes_gen]

    hmEpochs = len(learning_MSE_spikes_gen)
    last_spikes_mse = learning_MSE_spikes_gen[-1]
    return save_k, hmEpochs, last_spikes_mse


#all_tau_s = [3.0, 8.0, 10.0, 12.0, 15.0]
all_tau_s = [20.0, 30.0]
hm_calls = 100
what_to_teach = 'xor'
load = False
use_parallel = True
hm_workers = 20  # os.cpu_count()

meanEpochFromConverged = {}
mse_spikes = {}
ile_zbieglo = {}


# dla różnych tauów_s
for set_tau_s in all_tau_s:
    params = {'tau_syn_ex': set_tau_s,
              'tau_syn_in': set_tau_s}
    config = {'what_to_teach': what_to_teach}

    hmEpochs = []
    last_spikes_mse = []

    # puść hmCalls symulacji
    if use_parallel:
        if __name__ == '__main__':
            # obiekt odpowiedzialny za obliczenia rownoległe
            pe = parallel_for_online_learning(hm_workers, run)
            save, hmEpochs, last_spikes_mse = pe.evaluate(hm_calls, config, params)
    else:
        # bez obliczen równoległych
        save = []
        for k_call in range(hm_calls):
            save_k, hmEpochs_k, last_spikes_mse_k = run(k_call, params)

            hmEpochs.append(hmEpochs_k)
            last_spikes_mse.append(last_spikes_mse_k)
            save.append(save_k)

    meanEpochFromConverged[set_tau_s] = np.mean([x for x in hmEpochs if x < 500])
    ile_zbieglo[set_tau_s] = len([x for x in last_spikes_mse if x <= 0.25])

    filename = what_to_teach + '_result_' + str(set_tau_s)
    pickleData.save_object(save, filename)

# filename = 'rate_0.075_online_K2'
# foo = {'meanEpochFromConverged': meanEpochFromConverged,
#        'ile_zbieglo': ile_zbieglo}
# pickleData.save_object(foo, filename)
