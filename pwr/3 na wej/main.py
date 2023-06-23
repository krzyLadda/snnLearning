import nest
import matplotlib.pyplot as plt
import numpy as np
import nestShow as nsh
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

def reverse_scale(signal, bias, coeff):
    """
    Based on the given bias and coefficient,
    it determines the scaling of signals from [0, 1] range by:
    signal returned = (signal * coefficient) + bias

    Parameters
    ----------
    signal : float np.array
        Signal that will be re-scaled
    bias : float
        Bias = min value of original signal.
    coeff : float
        Scaling coefficient.

    Returns
    -------
    np.array
        Re-scaled signal.

    """
    return signal*coeff + bias


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


def run(task_id):
    """
    The main function of the main file. Its operation depends on global variables
    from the main file. Function created to allow execution of several learning in parralel
    (use parralel runs only for online learning!).

    Parameters
    ----------
    task_id : int
        ID of learning run.

    Returns
    -------
    save_k : list
        A list of things to be saved after learning as a result.
        See the code to know what is selected.

    """
    np.random.seed()
    nest.set_verbosity(18)  # don't show info from nest, warning are still visible
    # call config nest only once, when you start using it
    nestNet.config_nest()

    version = 'online_in_minibatches'  # version on learning
    hmSamples = None  # in minibatch!  # how many samples will be used during one epoch
    # how many proceses will be used, if batch or
    # online_in_minibaches version of learning is used
    hm_workers = 32
    # define that, networks will have connections from outputs to inputs while learning or not
    close_loop = True

    learning_rate = 0.05  # yep, this is learning rate
    # define at least one termination condition for learning
    # teaching will end if any of the following is met
    hmEpochs = 500  # maximal number of epochs
    # target value of MSE between target spikes times and network's outputs times
    # on learning data
    target_MSE_spikes = 0.003
    # target value of MSE between target singal values and network's outputs values
    target_learning_MSE = None
    # the same as target_MSE_spikes, but for test data - test data are not used in learning
    target_test_MSE = None

    if load:
        print('load')
        # load data from file
        load_data = pickleData.load_object(filename)
        # take appropriate network and other saved things
        net, learning_inputs, learning_targets, test_inputs, test_targets, config, \
            learning_MSE_gen, test_MSE_gen, learning_MSE_spikes_gen = load_data[task_id]

        hmIn = net.hmIn  # number of inputs to the network
        hmOut = net.hmOut  # number of outputs from the network
        hmN = net.hmN  # number of hidden neurons

    else:
        print('Start new learnig for pwr')
        hmIn = 2  # number of network's inputs
        hmOut = 1  # number of network's outputs
        hmN = 25  # number of hidden neurons

        # load learning data
        mat = scipy.io.loadmat('daneFinal.mat')
        learning_targets = mat["Pav"]
        learning_inputs = mat["x"]
        # I remove the first 10 samples because there is an error in them
        start = 10
        learning_inputs = learning_inputs[start:, :]
        learning_targets = learning_targets[start:, :]

        # scale the data to [0, 1] rangez
        learning_inputs, inputs_bias, inputs_coeff = scale(learning_inputs)
        learning_targets, targets_bias, targets_coeff = scale(learning_targets)

        # plot learning data
        plt.plot(learning_targets)
        plt.plot(learning_inputs)
        plt.show()

        # load test data, they should be scaled latter,
        # TODO: this was not used and not tested!!
        test_inputs = None
        test_targets = None

        # write in config to save in file later
        config = {}
        config['inputs_bias'] = inputs_bias
        config['inputs_coeff'] = inputs_coeff
        config['targets_bias'] = targets_bias
        config['targets_coeff'] = targets_coeff

        # create new network
        learning_MSE_gen = []
        test_MSE_gen = []
        learning_MSE_spikes_gen = []
        net = nestNet(hmIn, hmOut, hmN=hmN)

    # write in config to save in file later
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

    # start learning
    net.train(learning_inputs, learning_targets, learning_rate, hmEpochs=hmEpochs,
              target_learning_MSE=target_learning_MSE,
              target_test_MSE=target_test_MSE,
              test_inputs=test_inputs, test_targets=test_targets,
              learning_MSE_gen=learning_MSE_gen, test_MSE_gen=test_MSE_gen,
              task_id=task_id, dynamic=True, version=version, hm_workers=hm_workers,
              target_MSE_spikes=target_MSE_spikes,
              learning_MSE_spikes_gen=learning_MSE_spikes_gen, hmSamples=hmSamples,
              close_loop=close_loop)
    # save and return chosen
    save_k = [net, learning_inputs, learning_targets, test_inputs, test_targets, config,
              learning_MSE_gen, test_MSE_gen, learning_MSE_spikes_gen]
    return save_k


# how many times to perform learning from the beginning or which saved networks from "filename" file
# will be learn further, this is not connected with epochs!
hm_calls = list(range(10))  # [2, 3, 4]
filename = 'pwr_onlineInMinibatches_rate0.1_3wej'  # name of file, where results will be saved or from loaded
# if load == True -> contuunuate teaching from a saved file,
# False - start learning from the beginning
load = True
use_parallel = False  # flags, call the run function in parallel? True only for online learning!
# if use_parralel == True, then this defines how many procceses will be used
hmWorkers = 30  # os.cpu_count()

if use_parallel:
    if __name__ == '__main__':
        pe = parallel_run(hmWorkers, run)
        save = pe.evaluate()
else:
    save = []
    for k_call in hm_calls:
        save_k = run(k_call)
        save.append(save_k)
# save to file
pickleData.save_object(save, filename)
