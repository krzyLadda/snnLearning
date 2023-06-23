import nest
import matplotlib.pyplot as plt
import numpy as np
import nestShow as nsh
from nestNeuralNet import nestNet
import math
import pickleData

nest.set_verbosity(18)  # nie pokazuj info od nest

nestNet.config_nest()

what_to_teach = "interpolated_xor"

filename = what_to_teach + '_result'
data = pickleData.load_object(filename)

if what_to_teach == "interpolated_xor":
    net, learning_inputs, learning_targets, test_inputs, test_targets, config,\
           learning_MSE_gen, test_MSE_gen = data[0]
    # plot target
    x = config['inputs'][:, 0]
    y = config['inputs'][:, 1]
    z = config['target'].reshape(-1)
    fig = plt.figure(1)
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
    ax.view_init(25, -55)
    plt.show()
    for i in range(0, 10):
        net, learning_inputs, learning_targets, test_inputs, test_targets, config,\
               learning_MSE_gen, test_MSE_gen = data[i]
        # plot odpowiedzi sieci
        x = learning_inputs[:, 0]
        y = learning_inputs[:, 1]
        z = np.zeros(learning_targets.shape[0])
        for idx, t_in in enumerate(learning_inputs):
            a, *_ = net.respond(t_in.reshape(-1, 1))
            z[idx] = a[0]
        fig = plt.figure(i+2)
        ax = plt.axes(projection='3d')
        ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True, alpha=1.0)
        ax.view_init(25, -55)

        # dodaj testowe
        x = test_inputs[:, 0]
        y = test_inputs[:, 1]
        z = np.zeros(test_targets.shape[0])
        for idx, t_in in enumerate(test_inputs):
            a, *_ = net.respond(t_in.reshape(-1, 1))
            z[idx] = a[0]
        ax.scatter(x, y, z, marker='o', c='r')
        plt.show()

        # przebieg mse uczenia
        fig = plt.figure(100+i)
        plt.ylim(top=0.15)  # max([x for x in learning_MSE_gen if x < 1]))
        plt.plot(learning_MSE_gen)
        plt.show()

        # przebieg test uczenia
        fig = plt.figure(200+i)
        plt.ylim(top=0.15)  # max([x for x in test_MSE_gen if x < 1]))
        plt.plot(test_MSE_gen)
        plt.show()
