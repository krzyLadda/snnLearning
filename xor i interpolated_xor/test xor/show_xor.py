import nest
import matplotlib.pyplot as plt
import numpy as np
import nestShow as nsh
from nestNeuralNet import nestNet
import math
import pickleData

nest.set_verbosity(18)  # nie pokazuj info od nest

nestNet.config_nest()

what_to_teach = "xor"

filename = what_to_teach + '_result'
data = pickleData.load_object(filename)
hmData = len(data)
last_mse = []
last_spikes_mse = []
hmEpochs = []
for i in range(hmData):
    net, learning_inputs, learning_targets, test_inputs, test_targets, config,\
        learning_MSE_gen, test_MSE_gen, learning_MSE_spikes_gen = data[i][0]
    answer = net.respond_open_loop(learning_inputs)
    print(f'Target: \n{learning_targets},\n Respond: \n{answer}')

    # # mse
    # fig, ax = plt.subplots()
    # # target
    # plt.plot(learning_MSE_spikes_gen)
    # plt.show()

    # learning_MSE_spikes_gen = [0.5 * m for m in learning_MSE_spikes_gen]
    # learning_MSE_spikes_gen = [m for m in learning_MSE_spikes_gen if m > 0.5]

    hmEpochs.append(len(learning_MSE_spikes_gen))
    last_mse.append(net.mse(answer, learning_targets))
    last_spikes_mse.append(learning_MSE_spikes_gen[-1])
mean_last_mse = np.mean([x for x in last_mse if x < 1])
print(f'Mean_last_mse: {mean_last_mse}')

ile_zbiegło = len([x for x in last_spikes_mse if x <= 0.25])
meanEpochFromConverged = np.mean([x for x in hmEpochs if x < 1000])
