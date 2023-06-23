import nest
import matplotlib.pyplot as plt
import numpy as np
from nestNeuralNet import nestNet
import pickleData
# from myOde import myOde
from parallel import parallel_run
import os
import scipy.io

def calc_mse(target, x):
    return ((x - target)**2).mean()

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

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

filename = 'pwr_result_online_in_minibatch_close_loop_2sNaWej_rate0.1_nets_answers_and_mse'
odp_osobnikow = pickleData.load_object(filename)

mse = []
mse_ver1 = []
mse_ver2 = []

for i in range(len(odp_osobnikow[0])):
    learning_odp = odp_osobnikow[0][i]
    ver1_odp = odp_osobnikow[1][i]
    ver2_odp = odp_osobnikow[2][i]


    # plot learning data
    plt.figure(1)
    plt.plot(learning_odp, colors[i], label='Network '+str(i+1))
    mse.append(calc_mse(learning_odp, learning_targets))

    plt.figure(2)
    plt.plot(ver1_odp,  colors[i], label='Network '+str(i+1))
    mse_ver1.append(calc_mse(ver1_odp, test_targets_1[:]))

    plt.figure(3)
    plt.plot(ver2_odp,  colors[i], label='Network '+str(i+1))
    mse_ver2.append(calc_mse(ver2_odp, test_targets_2[:]))

plt.figure(1)
plt.plot(learning_targets, 'b--', label=r'Target trajectory', linewidth=1.0)
ax = plt.gca()
ax.set_ylabel(r'Average thermal power of the PWR $[\mathrm{W}]$')
ax.set_xlabel(r'$t[\mathrm{s}]$')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.40), fancybox=True, shadow=True, ncol=3)
plt.xlim(0, 1090)

plt.figure(2)
ax = plt.gca()
plt.plot(test_targets_1, 'b--', label=r'Target trajectory', linewidth=1.0)
ax.set_ylabel(r'Average thermal power of the PWR $[\mathrm{W}]$')
ax.set_xlabel(r'$t[\mathrm{s}]$')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.40), fancybox=True, shadow=True, ncol=3)
plt.xlim(0, len(test_targets_1))

plt.figure(3)
ax = plt.gca()
plt.plot(test_targets_2, 'b--', label=r'Target trajectory', linewidth=1.0)
ax.set_ylabel(r'Average thermal power of the PWR $[\mathrm{W}]$')
ax.set_xlabel(r'$t[\mathrm{s}]$')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.40), fancybox=True, shadow=True, ncol=3)
plt.xlim(0, len(test_targets_2))

plt.show()
