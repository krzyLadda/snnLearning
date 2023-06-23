import nest
import matplotlib.pyplot as plt
import numpy as np
from nestNeuralNet import nestNet
import math
import pickleData
import parallel
import os
from myOde import myOde
import scipy.io

def scale(signals, bias=None, coeff=None):
    if bias is not None and coeff is not None:
        have_biases_and_coeff = True
    else:
        have_biases_and_coeff = False
    # skaluje sygnaly do zakresu 0-1
    s = np.zeros(signals.shape)
    # ile wejsc do modelu
    hmSignals = signals.shape[1]
    # obróbka wejsc
    if not have_biases_and_coeff:
        coeff = np.zeros(hmSignals)
        bias = np.zeros(hmSignals)
    for i in range(hmSignals):
        col = signals[:, i]
        if not have_biases_and_coeff:
            bias[i] = min(col)
        col = col - bias[i]
        if not have_biases_and_coeff:
            coeff[i] = max(col) or 1
        s[:, i] = col / coeff[i]
    return s, bias, coeff


def run(task_id):
    np.random.seed()
    nest.set_verbosity(18)  # nie pokazuj info od nest
    nestNet.config_nest()

    load_net = None
    if load:
        print("load")
        filename = what_to_teach + '_result'
        load_data = pickleData.load_object(filename)
        net, learning_inputs, learning_targets, test_inputs, test_targets, config,\
            learning_MSE_gen, test_MSE_gen, learning_MSE_spikes_gen = load_data[task_id]

        hmIn = net.hmIn
        hmOut = net.hmOut
        hmN = net.hmN
        learning_rate = 0.01 # config['learning_rate']

        hmEpochs = 100
        target_MSE_spikes = None
        target_learning_MSE = None
        target_test_MSE = None

        version = 'batch'
        hmSamples = 289  # in minibatch!
        hm_workers = 32

        x0 = config['x0']

    elif what_to_teach == "M2-observer":

        print('M2-observer')
        hmN = 300

        version = 'batch'
        hmSamples = 289  # in minibatch!
        hm_workers = 32

        learning_rate = 0.1

        hmEpochs = 100
        target_MSE_spikes = 0.25
        target_learning_MSE = None
        target_test_MSE = None

        # wczytaj dane
        """ZREDUKOWANY ASM1 - model2 - obserwator"""
        # dane z modelu
        data = scipy.io.loadmat('learning_data.mat')
        learning_inputs = data['learning_inputs']
        # obróbka wejsc uczących
        learning_inputs, inputs_bias, inputs_coeff = scale(learning_inputs)
        learning_inputs = learning_inputs[0:289, :]

        learning_targets = data['learning_targets']
        # skalowanie wyjsc
        learning_targets, targets_bias, targets_coeff = scale(learning_targets)
        # learning_targets = learning_targets[0:289, 1].reshape(-1, 1)

        # ile wejsc do modelu
        hmU = learning_inputs.shape[1]
        # ile wyjsc z modelu
        hmOut = learning_targets.shape[1]
        # ile wejsc do sieci
        hmIn = hmU + hmOut

        # stan początkowy - wyjścia z sieci w chwili 0
        x0 = learning_targets[0]

        # dane testowe
        test_data = scipy.io.loadmat('test_data.mat')
        test_inputs = test_data['test_inputs']
        test_targets = test_data['test_targets']

        test_inputs, *_ = scale(test_inputs, bias=inputs_bias, coeff=inputs_coeff)
        test_targets, *_ = scale(test_targets, bias=targets_bias, coeff=targets_coeff)

        # przygotuj wejścia uczące jako wektor [u, x]
        # shifted_x = np.concatenate((learning_targets[0].reshape(1, -1), learning_targets))
        # shifted_x = shifted_x[:len(learning_inputs)]
        # learning_inputs = np.concatenate((learning_inputs, shifted_x), axis=1)  # x[0]

        config = {}
        config['inputs_bias'] = inputs_bias
        config['inputs_coeff'] = inputs_coeff
        config['targets_bias'] = targets_bias
        config['targets_coeff'] = targets_coeff
        config['hmSamples_in_batch'] = hmSamples
        config['x0'] = x0

    # do zapis
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
        raise 'Do not use two parallels at once.'

    # stworz lub wycztaj sieć
    if not load:
        learning_MSE_gen = []
        test_MSE_gen = []
        learning_MSE_spikes_gen = []
        net = nestNet(hmIn, hmOut, hmN=hmN)

    net.train(learning_inputs, learning_targets, learning_rate, hmEpochs=hmEpochs,
              target_learning_MSE=target_learning_MSE,
              target_test_MSE=target_test_MSE,
              test_inputs=test_inputs, test_targets=test_targets,
              learning_MSE_gen=learning_MSE_gen, test_MSE_gen=test_MSE_gen,
              x0=x0, task_id=task_id, dynamic=True, version=version, hm_workers=hm_workers,
              target_MSE_spikes=target_MSE_spikes,
              learning_MSE_spikes_gen=learning_MSE_spikes_gen, hmSamples=hmSamples)

    save_k = [net, learning_inputs, learning_targets, test_inputs, test_targets, config,
              learning_MSE_gen, test_MSE_gen, learning_MSE_spikes_gen]
    return save_k


hm_calls = 1
what_to_teach = 'M2-observer'
load = False
use_parallel = False
hm_workers = hm_calls  # os.cpu_count()

config = {'what_to_teach': what_to_teach}
# start tastks in background
if use_parallel:
    if __name__ == '__main__':
        # obiekt odpowiedzialny za obliczenia rownoległe
        pe = parallel.ParallelEvaluator(hm_workers, run)
        save = pe.evaluate(hm_calls, config)
else:
    # bez obliczen równoległych
    save = []
    for k_call in range(hm_calls):
        save_k = run(k_call)
        save.append(save_k)

filename = what_to_teach + '_result'
pickleData.save_object(save, filename)
