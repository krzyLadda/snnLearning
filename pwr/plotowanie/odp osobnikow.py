import nest
import matplotlib.pyplot as plt
import numpy as np
from nestNeuralNet import nestNet
import pickleData
# from myOde import myOde
from parallel import parallel_run
import os
import scipy.io

plt.rcParams.update({
    "text.usetex": True,
    # "font.family": "sans-serif",
    # "font.sans-serif": "Helvetica",
    'font.size': 11,
})
plt.rc('lines', linewidth=0.7)

# load learning data
mat = scipy.io.loadmat('daneFinal.mat')
learning_targets = mat["Pav"]
learning_inputs = mat["x"]
start = 10
learning_targets = learning_targets[start:, :]

mat = scipy.io.loadmat('testowy1.mat')
test_targets_1 = mat['Pav']

mat = scipy.io.loadmat('testowy2.mat')
test_targets_2 = mat['Pav']

filename = 'pwr_result_batch_open_loop_10sieci 2_nets_answers_and_mse'
odp_osobnikow = pickleData.load_object(filename)

for i in [0]:  # range(len(odp_osobnikow[0])):
    learning_odp = odp_osobnikow[0][i]
    ver1_odp = odp_osobnikow[1][i]
    ver2_odp = odp_osobnikow[2][i]

    # plot learning data
    plt.figure(i+0)
    ax = plt.gca()
    plt.plot(learning_targets, 'b--', label=r'Target trajectory')
    plt.plot(learning_odp, 'r', label=r'Response trajectory')
    ax.set_ylabel(r'Average thermal power of the PWR $[\mathrm{W}]$', size=14)
    ax.set_xlabel(r'$t[\mathrm{s}]$', size=14)
    ax.legend()
    plt.xlim(0, 1090)

    plt.figure(i+1)
    ax = plt.gca()
    plt.plot(test_targets_1, 'b--', label=r'Target trajectory')
    plt.plot(ver1_odp, 'r', label=r'Response trajectory')
    ax.set_ylabel(r'Average thermal power of the PWR $[\mathrm{W}]$', size=14)
    ax.set_xlabel(r'$t[\mathrm{s}]$', size=14)
    ax.legend()
    plt.xlim(0, len(ver1_odp))

    plt.figure(i+2)
    ax = plt.gca()
    plt.plot(test_targets_2, 'b--', label=r'Target trajectory')
    plt.plot(ver2_odp, 'r', label=r'Response trajectory')
    ax.set_ylabel(r'Average thermal power of the PWR $[\mathrm{W}]$', size=14)
    ax.set_xlabel(r'$t[\mathrm{s}]$', size=14)
    ax.legend()
    plt.xlim(0, len(ver2_odp))

    plt.show()
