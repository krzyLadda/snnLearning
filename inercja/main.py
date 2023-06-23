import nest
import matplotlib.pyplot as plt
import numpy as np
from nestNeuralNet import nestNet
import math
import pickleData
import parallel
import os
from myOde import myOde


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


def run(task_id):
    np.random.seed()
    nest.set_verbosity(18)  # nie pokazuj info od nest
    nestNet.config_nest(0.1)

    load_net = None
    if load:
        print("load")
        filename = what_to_teach + '_result'
        load_data = pickleData.load_object(filename)
        load_net, learning_inputs, learning_targets, test_inputs, test_targets, config, \
            learning_MSE_gen, test_MSE_gen = load_data[task_id]

        hmIn = load_net.hmIn
        hmOut = load_net.hmOut
        hmN_positive = load_net.hmN_positive
        hmN_negative = load_net.hmN_negative
        learning_rate = config['learning_rate']

        hmEpochs = 1000
        target_learning_MSE = None
        target_test_MSE = None  # 0.0025
        version = 'batch'
        if version == 'batch':
            hm_workers = 32
        else:
            hm_workers = 1

    elif what_to_teach == "first_order_inertia":
        print("first order intertia")
        # tutaj dane wejsciowe mało sie zmieniają wiec warto zmniejszyc dt symulatora
        hmIn = 2
        hmOut = 1
        hmN_positive = 20
        hmN_negative = 20

        learning_rate = 0.005

        hmEpochs = 1000
        target_learning_MSE = None
        target_test_MSE = None  # 0.0025
        version = 'online'
        if version == 'batch':
            hm_workers = 32
        else:
            hm_workers = 1

        # obiekt / model
        dt = 0.1

        def f(t, x, u):
            a = -0.25
            b = 1
            return a*x + b*u

        # wejścia uczące
        u0 = np.sin(np.linspace(0, np.pi*5, 300))
        u1 = np.full(300, -1)
        u2 = np.full(300, 0.5)
        u3 = np.full(300, 1)
        learning_inputs = np.concatenate((u0, u1, u2, u3)).reshape(-1, 1)

        x0 = 0
        learning_targets = np.zeros(learning_inputs.shape)
        model = myOde(f, dt, x0, 0, 'rk4')
        # wyjście uczące
        for idx, u_k in enumerate(learning_inputs):
            learning_targets[idx] = model.update(u_k)  # x[1] x[2]
        # skalowanie uczacych
        learning_inputs, inputs_bias, inputs_coeff = scale(learning_inputs)
        learning_targets, targets_bias, targets_coeff = scale(learning_targets)

        shifted_x = np.concatenate((np.array(0, ndmin=2), learning_targets))
        learning_inputs = np.concatenate(
            (learning_inputs.reshape(-1, 1), shifted_x[:len(learning_inputs)].reshape(-1, 1)),
            axis=1)  # x[0]

        # testowe
        u0 = np.full(200, -0.5)
        u1 = np.full(200, 0.5)
        u2 = np.linspace(0.5, 1, 150)
        u3 = np.linspace(1, 0.5, 150)
        u4 = 0.5 - np.sin(np.linspace(0, np.pi, 100))
        test_inputs = np.concatenate((u0, u1, u2, u3, u4)).reshape(-1, 1)
        # wyjscia testowe
        model = myOde(f, dt, x0, 0, 'rk4')
        test_targets = np.zeros(test_inputs.shape)
        for idx, u_k in enumerate(test_inputs):
            test_targets[idx] = model.update(u_k)  # x[1] x[2]
        # skalowanie testowych
        test_inputs = scale(test_inputs, inputs_bias, inputs_coeff)
        test_targets = scale(test_targets, targets_bias, targets_coeff)

        config = {}
        config['inputs_bias'] = inputs_bias
        config['inputs_coeff'] = inputs_coeff
        config['targets_bias'] = targets_bias
        config['targets_coeff'] = targets_coeff

    # do zapis
    config['what_to_teach'] = what_to_teach
    config['learning_inputs'] = learning_inputs
    config['learning_targets'] = learning_targets
    config['test_inputs'] = test_inputs
    config['test_targets'] = test_targets
    config['hmIn'] = hmIn
    config['hmOut'] = hmOut
    config['hmN_positive'] = hmN_positive
    config['hmN_negative'] = hmN_negative
    config['learning_rate'] = learning_rate
    config['hmEpochs'] = hmEpochs
    config['target_learning_MSE'] = target_learning_MSE
    config['target_test_MSE'] = target_test_MSE
    config['x0'] = x0
    config['version'] = version

    if version == 'batch' and use_parallel:
        raise 'Nie używaj dwóch równoległości na raz'

    # stworz lub wycztaj sieć
    if load_net is not None:
        net = load_net
    else:
        learning_MSE_gen = []
        test_MSE_gen = []
        net = nestNet(hmIn, hmOut, hmN_positive=hmN_positive, hmN_negative=hmN_negative)

    net.train(learning_inputs, learning_targets, learning_rate, hmEpochs=hmEpochs,
              target_learning_MSE=target_learning_MSE,
              target_test_MSE=target_test_MSE,
              test_inputs=test_inputs, test_targets=test_targets,
              learning_MSE_gen=learning_MSE_gen, test_MSE_gen=test_MSE_gen,
              x0=x0, task_id=task_id, dynamic=True, version=version, hm_workers=hm_workers)

    save_k = [net, learning_inputs, learning_targets, test_inputs, test_targets, config,
              learning_MSE_gen, test_MSE_gen]
    return save_k


hm_calls = 10
what_to_teach = 'first_order_inertia'
load = False
use_parallel = False
hm_workers = hm_calls  # os.cpu_count()

config = {'what_to_teach': what_to_teach}
# start tastks in background
if use_parallel:
    if __name__ == '__main__':
        # obiekt odpowiedzialny za obliczenia rownoległe
        pe = parallel.ParallelEvaluator(hm_workers, run)
        save = pe.evaluate(hm_calls, config)
else:
    # bez obliczen równoległych
    save = []
    for k_call in range(hm_calls):
        save_k = run(k_call)
        save.append(save_k)

filename = what_to_teach + '_result'
pickleData.save_object(save, filename)
