import nest
import matplotlib.pyplot as plt
import numpy as np
import nestShow as nsh
from nestNeuralNet import nestNet
import math
import pickleData
import scipy.io


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

plt.rcParams.update({
    "text.usetex": True,
    # "font.family": "sans-serif",
    # "font.sans-serif": "Helvetica",
    'font.size': 11,
})
plt.rc('lines', linewidth=0.7)

np.random.seed()
nest.set_verbosity(18)  # nie pokazuj info od nest
nestNet.config_nest()

filename = 'pwr_result_online_in_minibatch_close_loop 4'
data = pickleData.load_object(filename)

what_to_show = ['learning', 'testowe1', 'testowe2']  # 'learning',\

plt.rcParams.update({
    "text.usetex": True,
    # "font.family": "sans-serif",
    # "font.sans-serif": "Helvetica",
    'font.size': 11,
})
plt.rc('lines', linewidth=0.7)



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


def plot_and_calc(data_filename, config, net):
    mat = scipy.io.loadmat(data_filename)
    test_inputs = mat['x']
    rescaled_test_targets = mat['Pav']


    inputs_bias = config['inputs_bias']
    inputs_coeff = config['inputs_coeff']
    test_inputs, *_ = scale(test_inputs, inputs_bias, inputs_coeff)

    targets_bias = config['targets_bias']
    targets_coeff = config['targets_coeff']
    test_targets, *_ = scale(rescaled_test_targets, targets_bias, targets_coeff)

    x0 = test_targets[0]
    test_targets = test_targets[1:]
    test_inputs = test_inputs[:test_targets.shape[0]]

    answer = net.respond_close_loop(test_inputs, x0)
    rescaled_answer = reverse_scale(answer, config['targets_bias'], config['targets_coeff'])
    mse = net.mse(answer, test_targets)

    hmSamples = len(test_targets)
    t = np.arange(0, hmSamples, 1)
    fig, ax = plt.subplots()
    # target
    plt.plot(t, rescaled_test_targets[1:], 'b--', label='Reference trajectory')
    # odpowiedź
    plt.plot(t, rescaled_answer , 'r', label='Network response')
    plt.legend()
    plt.show()

    return rescaled_answer, mse

mse = []
mse_ver1 = []
mse_ver2 = []

learning_odp = []
ver1_odp = []
ver2_odp = []

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
        x0 = learning_targets[0]

        answer_close = net.respond_close_loop(learning_inputs, x0)
        learning_odp.append(reverse_scale(answer_close, config['targets_bias'],
                                          config['targets_coeff']))
        mse_learning_close_loop = net.mse(answer_close, learning_targets)
        mse.append(mse_learning_close_loop)
        print(f"mse learning in close loop = {mse_learning_close_loop}")
        t = np.arange(0, hmSamples, 1)
        fig, ax = plt.subplots()
        # target
        plt.plot(t, learning_targets, 'b--', label='Reference trajectory')
        # odpowiedź
        plt.plot(answer_close, 'r', label='Network response')
        plt.legend()
        plt.show()

    # %% plot odpowiedzi sieci testowe 1
    if 'testowe1' in what_to_show:
        answer_ver1, temp_mse_ver1 = plot_and_calc('testowy1.mat', config, net)
        ver1_odp.append(answer_ver1)
        mse_ver1.append(temp_mse_ver1)

    # %% plot odpowiedzi sieci testowe 2
    if 'testowe2' in what_to_show:
        answer_ver2, temp_mse_ver2 = plot_and_calc('testowy2.mat', config, net)
        ver2_odp.append(answer_ver2)
        mse_ver2.append(temp_mse_ver2)


save = [learning_odp, ver1_odp, ver2_odp, mse, mse_ver1, mse_ver2]
filename_save = filename + '_nets_answers_and_mse'
pickleData.save_object(save, filename_save)
