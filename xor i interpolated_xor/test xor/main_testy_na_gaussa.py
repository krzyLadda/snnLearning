import nest
import matplotlib.pyplot as plt
import numpy as np
from nestNeuralNet import nestNet
import math
import pickleData
from parallel import parallel_for_online_learning
import os
from scipy import optimize


def run(task_id, params=None):
    np.random.seed()
    nest.set_verbosity(18)  # nie pokazuj info od nest
    nestNet.config_nest(params)

    hmIn = 3
    hmOut = 1
    hmN = 5

    version = 'batch'
    use_parallel = False
    hm_workers = 4

    hmEpochs = 100
    target_MSE_spikes = 0.25
    target_learning_MSE = None
    target_test_MSE = None

    learning_rate = 0.25  # + 0.0009*np.exp(-np.arange(0, hmEpochs, 1) / 300)

    learning_inputs = [[1, 1, 1],  # 1 1
                       [0, 1, 1],  # 0 1
                       [1, 0, 1],  # 1 0
                       [0, 0, 1]]  # 0 0
    learning_inputs = np.array(learning_inputs)

    learning_targets = [[0],  # 0
                        [1],  # 1
                        [1],  # 1
                        [0]]  # 0
    learning_targets = np.array(learning_targets)

    test_inputs = None
    test_targets = None

    config = {}

    # do zapisu
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
        raise 'Do not use two parralels in one time.'

    # stworz sieć
    learning_MSE_gen = []
    test_MSE_gen = []
    learning_MSE_spikes_gen = []
    net = nestNet(hmIn, hmOut, hmN=hmN)

    net.train(learning_inputs, learning_targets, learning_rate, hmEpochs=hmEpochs,
              target_learning_MSE=target_learning_MSE,
              target_test_MSE=target_test_MSE,
              test_inputs=test_inputs, test_targets=test_targets,
              learning_MSE_gen=learning_MSE_gen, test_MSE_gen=test_MSE_gen,
              task_id=task_id, dynamic=False, version=version, hm_workers=hm_workers,
              learning_MSE_spikes_gen=learning_MSE_spikes_gen, target_MSE_spikes=target_MSE_spikes)

    hmEpochs = len(learning_MSE_spikes_gen)
    return None, hmEpochs, None


def opt_f(x):
    print(f"Wywołanie funkcji z paramterami {x}, warunek jest spelniony? {x[1]>=x[0]}")

    hm_calls = 30
    use_parallel = False
    hm_workers = 7   # os.cpu_count()

    params = {'hidden_weights_mean': x[0],
              'output_weights_mean': x[1],
              'hidden_weights_std': x[2],
              'output_weights_std': x[2]  # 'output_weights_std': x[3]
              }

    config = {}
    hmEpochs = []
    last_spikes_mse = []

    # puść hmCalls symulacji
    if use_parallel:
        if __name__ == '__main__':
            # obiekt odpowiedzialny za obliczenia rownoległe
            pe = parallel_for_online_learning(hm_workers, run)
            save, hmEpochs, last_spikes_mse = pe.evaluate(hm_calls, config, params)
    else:
        # bez obliczen równoległych
        for k_call in range(hm_calls):
            save_k, hmEpochs_k, last_spikes_mse_k = run(k_call, params)

            hmEpochs.append(hmEpochs_k)
            # last_spikes_mse.append(last_spikes_mse_k)
            # save.append(save_k)
    r = np.mean(hmEpochs)
    # points[tuple(x)] = r
    print(f"Wynik:{r}")
    return r


# brute force
points = {}

mean_hidden = np.arange(0.5, 10.1, 0.5)
mean_output = np.arange(0.5, 10.1, 0.5)
std_both = np.arange(0.5, 5.1, 0.5)

hm_iter = len(mean_hidden) * len(mean_output) * len(std_both)
i = 1
for m_h in mean_hidden:
    for m_o in mean_output:
        # jeżeli mean hidden jest wieksze od mean output to nie rób
        i = i+1
        if m_h > m_o:
            continue
        for std in std_both:
            print(f"Iteracaja {i} z {hm_iter} -> wykonano {i*100/hm_iter}%.")
            x = [m_h, m_o, std]
            points[tuple(x)] = opt_f(x)


filename = 'brute_force_K5_batch_rate0.25'
foo = {'points': points}
pickleData.save_object(foo, filename)


######################################################################
# points = {}
# def con1(x):
#     return x[1]-x[0] # >=0

# cons = ({'type': 'ineq', 'fun': con1})
# bounds = [(0.1, 10), (0.1, 10), (0.5, 5), (0.5, 5)]
# options = {'f_tol': 2.0}
# #res = optimize.differential_evolution(opt_f, bounds, polish=True,
# #                                      constraints=cons, x0=[0.5, 2.0, 1.0, 1.0])
# res = optimize.shgo(opt_f, bounds, options=options, constraints=cons, n=100, iters=4)


# filename = 'rate_0.175_K1_opty'
# foo = {'opt_result': res}
# pickleData.save_object(foo, filename)
