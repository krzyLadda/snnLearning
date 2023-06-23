import nest
import matplotlib.pyplot as plt
import numpy as np
import nestShow as nsh
from nestNeuralNet_2sNaWej import nestNet
import math
import pickleData
from myOde import myOde
from parallel import parallel_run
import os
import scipy.io


filename = '14 pwr_result_online_in_minibatch_close_loop_2sNaWej_rate0.01'  # rate 0.01

def reverse_scale(signal, bias, coeff):
    return signal*coeff + bias


def scale(signals, bias=None, coeff=None):
    if bias and coeff:
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

    what_to_teach = 'pwr'

    load_net = None
    if load:
        print('load')
        load_data = pickleData.load_object(filename)
        net, learning_inputs, learning_targets, test_inputs, test_targets, config, \
            learning_MSE_gen, test_MSE_gen, learning_MSE_spikes_gen = load_data[task_id]

        hmIn = net.hmIn
        hmOut = net.hmOut
        hmN = net.hmN

        version = 'online_in_minibatches'
        hmSamples = None # in minibatch!
        hm_workers = 32
        close_loop = True

        learning_rate = 0.01 # config['learning_rate']

        hmEpochs = 200
        target_MSE_spikes = 0.001
        target_learning_MSE = None
        target_test_MSE = None

    elif what_to_teach == "pwr":
        print('pwr')
        # tutaj dane wejsciowe mało sie zmieniają wiec warto zmniejszyc dt symulatora
        hmIn = 2
        hmOut = 1
        hmN = 25

        version = 'online_in_minibatches'
        hmSamples = None #1100  # in minibatch!
        hm_workers = 32
        close_loop = True

        learning_rate = 0.1

        hmEpochs = 500
        target_MSE_spikes = 0.005
        target_learning_MSE = None
        target_test_MSE = None

        mat = scipy.io.loadmat('daneFinal.mat')
        learning_targets = mat["Pav"]
        learning_inputs = mat["x"]
        start = 10
        learning_inputs = learning_inputs[start:, :]
        learning_targets = learning_targets[start:, :]

        # skalowanie danych
        learning_inputs, inputs_bias, inputs_coeff = scale(learning_inputs)
        learning_targets, targets_bias, targets_coeff = scale(learning_targets)

        learning_inputs
        plt.plot(learning_targets)
        plt.plot(learning_inputs)
        plt.show()

        test_inputs = None
        test_targets = None

        config = {}
        config['inputs_bias'] = inputs_bias
        config['inputs_coeff'] = inputs_coeff
        config['targets_bias'] = targets_bias
        config['targets_coeff'] = targets_coeff

    # do zapis
    config['close_loop'] = close_loop
    config['hmSamples_in_batch'] = hmSamples
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
              task_id=task_id, dynamic=True, version=version, hm_workers=hm_workers,
              target_MSE_spikes=target_MSE_spikes,
              learning_MSE_spikes_gen=learning_MSE_spikes_gen, hmSamples=hmSamples,
              close_loop=close_loop)

    save_k = [net, learning_inputs, learning_targets, test_inputs, test_targets, config,
              learning_MSE_gen, test_MSE_gen, learning_MSE_spikes_gen]
    return save_k


hm_calls = list(range(10)) #[2, 3, 4]
load = True
use_parallel = False
hmWorkers = 30  # os.cpu_count()

# start tastks in background
if use_parallel:
    if __name__ == '__main__':
        #obiekt odpowiedzialny za obliczenia rownoległe
        pe = parallel_run(hmWorkers, run)
        save = pe.evaluate()
else:
    # bez obliczen równoległych
    save = []
    for k_call in hm_calls:
        save_k = run(k_call)
        save.append(save_k)

pickleData.save_object(save, filename)
