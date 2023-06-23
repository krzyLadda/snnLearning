import nest
import matplotlib.pyplot as plt
import numpy as np
import nestShow as nsh
from nestNeuralNet import nestNet
import math
import pickleData

nest.set_verbosity(18)  # nie pokazuj info od nest

nestNet.config_nest()

what_to_teach = "M2-observer"

filename = what_to_teach + '_result'
data = pickleData.load_object(filename)
hmData = len(data)
last_mse = []
last_spikes_mse = []
hmEpochs = []
for i in range(hmData):
    net, learning_inputs, learning_targets, test_inputs, test_targets, config,\
           learning_MSE_gen, test_MSE_gen, learning_MSE_spikes_gen = data[i]
    learning_answer = net.respond_close_loop(learning_inputs, config['x0'])
    learning_targets = learning_targets[0:289, :]
    #test_answer = net.respond_close_loop(test_inputs, test_targets[0])

    # mse
    fig, ax = plt.subplots()
    # target
    plt.plot(learning_MSE_spikes_gen)
    plt.show()

    hmEpochs.append(len(learning_MSE_spikes_gen))
    #last_mse.append(net.mse(learning_answer, learning_targets))
    #last_spikes_mse.append(learning_MSE_spikes_gen[-1])
#mean_last_mse = np.mean([x for x in last_mse])
#print(f'Mean_last_mse: {mean_last_mse}')

ile_zbiegło = len([x for x in last_spikes_mse if x <= 0.5])
meanEpochFromConverged  = np.mean([x for x in hmEpochs if x<1000])

# %% plot odp
hmPlots = learning_targets.shape[1]
hmSamples = learning_targets.shape[0]
t = np.arange(0, hmSamples, 1)
for i in range(hmPlots):
    plt.figure(i+1)
    plt.plot(t, learning_targets[:, i], 'b--', label='Reference trajectory')
    plt.plot(t, learning_answer[:, i], 'r', label='Net respond')
    plt.legend()
    plt.show()


