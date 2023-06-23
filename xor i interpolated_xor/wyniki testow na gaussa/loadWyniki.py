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

filename = "brute_force_K5_batch_rate0.25"
data = pickleData.load_object(filename)

points = data['points']
min_params = dict([(k, v) for (k, v) in points.items() if v == min(points.values())])
print(min_params)
print("Mean hidden, mean output, std")
