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


def scale(signals, bias=None, coeff=None):
    """
    Scales signals to [0, 1] range, as in equations (24) and (25) in article.

    Parameters
    ----------
    signals : list of float np.arrays
        Signals that will be scaled.
    bias : list of floats, optional
        Bias = min value of original signal. If None will be calculated. The default is None.
    coeff : list of floats, optional
        Scaling coefficient. If None will be calculated. The default is None.

    Returns
    -------
    s : np.array
        Scaled signal.
    bias : list of floats
        Biases of signals.
    coeff : list of floats
        Scaling coefficients of signals.

    """
    if bias and coeff:
        have_biases_and_coeff = True
    else:
        have_biases_and_coeff = False
    s = np.zeros(signals.shape)
    hmSignals = signals.shape[1]
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


# load learning data
mat = scipy.io.loadmat('testowy2.mat')
learning_targets = mat["Pav"]
learning_inputs = mat["x"]
# I remove the first 10 samples because there is an error in them
start = 10
learning_inputs = learning_inputs[start:, :]
learning_targets = learning_targets[start:, :]

# # scale the data to [0, 1] rangez
# learning_inputs, inputs_bias, inputs_coeff = scale(learning_inputs)
# learning_targets, targets_bias, targets_coeff = scale(learning_targets)

# plot learning data
plt.figure(0)
ax = plt.gca()
plt.plot(learning_targets)
ax.set_ylabel(r'Average thermal power of the PWR $[\mathrm{W}]$', size=14)
ax.set_xlabel(r'$t[\mathrm{s}]$', size=14)
plt.xlim(0, 1090)

plt.figure(1)
ax = plt.gca()
plt.plot(learning_inputs)
ax.set_ylabel(r'Immersion of control rods in the PWR $[\mathrm{m}]$', size=14)
ax.set_xlabel(r'$t[\mathrm{s}]$', size=14)
plt.xlim(0, 1090)

plt.show()
