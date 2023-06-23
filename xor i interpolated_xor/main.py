import nest
import matplotlib.pyplot as plt
import numpy as np
from nestNeuralNet import nestNet
import math
import pickleData
from parallel import parallel_for_online_learning
import os


def run(task_id):
    np.random.seed()
    nest.set_verbosity(18)  # nie pokazuj info od nest
    nestNet.config_nest()

    load_net = None
    start_hm_epochs = 0
    if load:
        print('load')
        filename = what_to_teach + '_result'
        load_data = pickleData.load_object(filename)
        load_net, learning_inputs, learning_targets, test_inputs, test_targets, config, \
            learning_MSE_gen, test_MSE_gen, learning_MSE_spikes_gen = load_data[task_id]

        hmIn = load_net.hmIn
        hmOut = load_net.hmOut
        hmN = load_net.hmN
        start_hm_epochs = len(learning_MSE_spikes_gen)

        version = 'batch'
        hm_workers = 4
        learning_rate = 0.1  # config['learning_rate']

        hmEpochs = 1000
        target_MSE_spikes = 0.25
        target_learning_MSE = None
        target_test_MSE = None

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

        learning_rate = 0.01  # + 0.0009*np.exp(-np.arange(0, hmEpochs, 1) / 300)

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

    if len(learning_MSE_spikes_gen) - start_hm_epochs < hmEpochs:
        converged = True
    else:
        converged = False

    return save_k, converged


hm_calls = 10
what_to_teach = 'xor'
load = False
use_parallel = False
hm_workers = hm_calls  # os.cpu_count()

config = {'what_to_teach': what_to_teach}
# start tastks in background
if use_parallel:
    if __name__ == '__main__':
        # obiekt odpowiedzialny za obliczenia rownoległe
        pe = parallel_for_online_learning(hm_workers, run)
        save = pe.evaluate(hm_calls, config)
else:
    # bez obliczen równoległych
    save = []
    for k_call in range(hm_calls):
        converged = False
        while not converged:
            save_k, converged = run(k_call)
            save.append(save_k)

filename = what_to_teach + '_result'
pickleData.save_object(save, filename)
