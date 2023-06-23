import nest
import matplotlib.pyplot as plt
import numpy as np
import nestShow as nsh
from nestNeuralNet import nestNet
import math
import pickleData

nestNet.config_nest()
nest.set_verbosity(18)  # nie pokazuj info od nest

filename = 'first_order_inertia_result'
data = pickleData.load_object(filename)
for i in range(10):
    net, learning_inputs, learning_targets, test_inputs, test_targets, config,\
                   learning_MSE_gen, test_MSE_gen = data[i]

    # plot dane uczace
    x0 = 0
    inputs = learning_inputs[:, 0]
    answer = net.respond_close_loop(inputs, x0)

    hmSamples = len(learning_targets)
    t = np.arange(0, hmSamples, 1)
    fig, ax = plt.subplots()
    # target
    plt.plot(t, learning_targets, 'b--', label='Reference trajectory')
    # odpowiedź
    plt.plot(t, answer, 'r', label='Net respond')
    plt.show()

    # plot odpowiedzi sieci testowe 1
    x0 = 0
    answer = net.respond_close_loop(test_inputs, x0)

    hmSamples = len(test_targets)
    t = np.linspace(0, hmSamples, 1)
    fig, ax = plt.subplots()
    # target
    plt.plot(t, test_targets, 'b--', label='Reference trajectory')
    # odpowiedź
    plt.plot(t, answer, 'r', label='Net respond')
    plt.show()
