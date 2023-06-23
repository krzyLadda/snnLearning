import nest
import matplotlib.pyplot as plt
import numpy as np
import nestShow as nsh
from nestNeuralNet import nestNet
import math
import pickleData
import scipy.io

np.random.seed()
nest.set_verbosity(18)  # nie pokazuj info od nest
nestNet.config_nest()

filename = 'pwr_onlineInMinibatches_rate0.1_3wej'
data = pickleData.load_object(filename)

what_to_show = ['learning', 'testowe1', 'testowe2']  # 'learning',\

plt.rcParams.update({'font.size': 11})

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

mse = []
mse_ver1 = []
mse_ver2 = []

# %% MAIN
for i in range(10):
    net, learning_inputs, learning_targets, test_inputs, test_targets, config,\
        learning_MSE_gen, test_MSE_gen, learning_MSE_spikes_gen = data[i]

    # plt.plot(learning_targets)
    # plt.plot(learning_inputs)
    # plt.show()
    # idx_start = 400
    # learning_inputs = learning_inputs[idx_start:,:]
    # learning_targets = learning_targets[idx_start:,:]

    plt.plot(learning_MSE_spikes_gen[:])
    plt.show()
    # %% plot dane uczace
    # continue
    if 'learning' in what_to_show:
        hmSamples = learning_targets.shape[0]
        t = np.arange(0, hmSamples, 1)
        x0 = learning_targets[0]
        # # nowe przesuwanie
        # learning_inputs = np.concatenate((learning_inputs, learning_targets), axis=1)
        # learning_targets = learning_targets[1:]
        # inputs_open_loop = learning_inputs[:learning_targets.shape[0]]
        # answer_open = net.respond_open_loop(inputs_open_loop)
        # mse_learning_open_loop = net.mse(answer_open, learning_targets)
        # print(f"mse learning in open loop = {mse_learning_open_loop}")
        # fig, ax = plt.subplots()
        # # target
        # plt.plot(t[0:learning_targets.shape[0]], learning_targets, 'b--', label='Reference trajectory')
        # # odpowiedź
        # plt.plot(answer_open, 'r', label='Net respond - open loop')
        # plt.legend()
        # plt.show()

        answer_close = net.respond_close_loop(learning_inputs, x0)
        mse_learning_close_loop = net.mse(answer_close, learning_targets)
        mse.append(mse_learning_close_loop)
        print(f"mse learning in close loop = {mse_learning_close_loop}")
        fig, ax = plt.subplots()
        # target
        plt.plot(t, learning_targets, 'b--', label='Reference trajectory')
        # odpowiedź
        plt.plot(answer_close, 'r', label='Net respond - close loop')
        plt.legend()
        plt.show()

    # %% plot odpowiedzi sieci testowe 1
    if 'testowe1' in what_to_show:
        filename = 'testowy1.mat'
        mat = scipy.io.loadmat(filename)
        test_inputs = mat['x']
        test_targets = mat['Pav']

        inputs_bias = config['inputs_bias']
        inputs_coeff = config['inputs_coeff']
        test_inputs, *_ = scale(test_inputs, inputs_bias, inputs_coeff)

        targets_bias = config['targets_bias']
        targets_coeff = config['targets_coeff']
        test_targets, *_ = scale(test_targets, targets_bias, targets_coeff)

        x0 = test_targets[0]
        answer = net.respond_close_loop(test_inputs, x0)
        mse_ver1.append(net.mse(answer, test_inputs))

        hmSamples = len(test_targets)
        t = np.arange(0, hmSamples, 1)
        fig, ax = plt.subplots()
        # target
        plt.plot(t, test_targets, 'b--', label='Reference trajectory')
        # odpowiedź
        plt.plot(t, answer, 'r', label='Net respond')
        plt.legend()
        plt.show()

    # %% plot odpowiedzi sieci testowe 2
    if 'testowe2' in what_to_show:
        filename = 'testowy2.mat'
        mat = scipy.io.loadmat(filename)
        test_inputs = mat['x']
        test_targets = mat['Pav']

        inputs_bias = config['inputs_bias']
        inputs_coeff = config['inputs_coeff']
        test_inputs, *_ = scale(test_inputs, inputs_bias, inputs_coeff)

        targets_bias = config['targets_bias']
        targets_coeff = config['targets_coeff']

        test_targets, *_ = scale(test_targets, targets_bias, targets_coeff)

        x0 = test_targets[0]
        answer = net.respond_close_loop(test_inputs, x0)
        mse_ver2.append(net.mse(answer, test_inputs))

        hmSamples = len(test_targets)
        t = np.arange(0, hmSamples, 1)
        fig, ax = plt.subplots()
        # target
        plt.plot(t, test_targets, 'b--', label='Reference trajectory')
        # odpowiedź
        plt.plot(t, answer, 'r', label='Net respond')
        plt.legend()
        plt.show()
