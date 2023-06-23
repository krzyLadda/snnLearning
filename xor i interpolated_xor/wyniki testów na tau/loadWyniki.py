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

filename = "rate_0.5_testy_tau_s"
data = pickleData.load_object(filename)
