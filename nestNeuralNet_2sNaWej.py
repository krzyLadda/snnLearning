import numpy as np
import math
import graphs
import nest
import nestShow as nsh
from random import randint, random, uniform, shuffle, sample
import matplotlib.pyplot as plt
import copy
from parallel import parallel_for_learning

"""spikes are understand as pairs: neuron number: spike time"""

# what2show = ['I_syn_in', 'I_syn_ex', 'V_m', 'spikedetector.senders']

# flag of negative weights, True - weights only positive
weights_only_positive = False


weights_init = {'hidden_weights_mean': 0.5,
                'hidden_weights_std': 1.0,
                'output_weights_mean': 2.0,
                'output_weights_std': 1.0}

# define neuron model, so far only iaf_psc_alpha
neuronModel = "iaf_psc_alpha"
# model parameters
# TODO: requires greater flexibility in parametrisation
defaultModel = False
if defaultModel:
    IAF_PSC_EXP_PARAMS = nest.GetDefaults(neuronModel)
else:
    # assumption: tau_syn_ex == tau_syn_in
    IAF_PSC_EXP_PARAMS = {"E_L": 0.0, "V_th": 1.0, "V_reset": 0.0, "V_m": 0.0, "tau_m": 5.0,  # 1.0
                          "C_m": 5.0, 't_ref': 1000.0, 'tau_syn_ex': 10.0, 'tau_syn_in': 10.0}
    # tau syn ex 7.0 tau syn in 7.0


class nestNet:
    neuronsModel = None
    generatorModel = None
    dt = 0.01  # step time of nest simulation, default 0.1
    simTime = 20
    # simulaton samples are incres by: +1 for zero on the beginning
    simulationSamples = round(simTime/dt) + 1
    synapse_model = "static_synapse"
    # target spikes time range
    maxOutSpikeTime = 16 #14 #16
    minOutSpikeTime = 10 #7 #10
    diffOutSpikeTime = maxOutSpikeTime - minOutSpikeTime
    # inputs spikes time range
    minInSpikeTime = dt
    maxInSpikeTime = 6.0
    diffInSpikeTime = maxInSpikeTime - minInSpikeTime

    max_synapse_delay = 16

    def __init__(self, hmIn, hmOut, config=None, evolv_params=None, hmN=None):
        """ Initializes the net by:
            - saves number of inputs, outputs, hidden neurons, and their lists

            - create connections.

            Arguments:
            -------
                None.

            Returns
            -------
                None.
        """
        self.delays = np.arange(0, nestNet.max_synapse_delay+0.1, 1.0)
        # technically easier to make delay equal to time step than zero
        # I assumed that impact is negligible:
        self.delays[0] = nestNet.dt
        # print("delays:")
        # print(self.delays)

        self.hmOut = hmOut
        self.hmIn = hmIn + 2  # plus 2 for inputs that spikes allways on ,,0" (nestNet.dt) time,
                              # and at the middle
        if hmN is None:
            self.hmN = [randint(1, config["maxHiddenNeuronsInit"])]
        else:
            self.hmN = hmN

        # numbering of nodes is in order: input, output, hidden
        self.input_neurons = list(range(1, self.hmIn+1))
        foo = self.hmOut + self.hmIn + 1
        self.output_neurons = list(range(self.hmIn+1, foo))
        self.hidden_neurons = list(range(foo, foo+self.hmN))
        # save hidden neurons in layers (so far 1 layer)
        self.layers = [self.hidden_neurons]

        # create connections, all from n layer to all from n+1 layer
        self.connections = {}
        for d in self.delays:
            # connection from inputs to hidden
            for preSyn in self.input_neurons:
                for postSyn in self.hidden_neurons:
                    self.connections[preSyn, postSyn, d] =\
                        np.random.normal(loc=weights_init['hidden_weights_mean'],
                                         scale=weights_init['hidden_weights_std'])
            # connections from hidden to outpus
            for preSyn in self.hidden_neurons:
                for postSyn in self.output_neurons:
                    self.connections[preSyn, postSyn, d] =\
                        np.random.normal(loc=weights_init['output_weights_mean'],
                                         scale=weights_init['output_weights_std'])

        if weights_only_positive:
            self.connections = {k: abs(v) for k, v in self.connections.items()}

    @staticmethod
    def config_nest(params=None):
        """Configures nest.
        Call that function only once at the beginning of using NEST.

        Arguments:
        -------
            None.

        Returns
        -------
            None."""

        if nestNet.maxOutSpikeTime >= nestNet.simTime:
            raise "Error: simTime < max time of output target!"
        nest.ResetKernel()
        # set simulation step time
        nest.SetKernelStatus({"resolution": nestNet.dt})
        # depending on the selected type of neuron set its parameters and
        # calculate neuron outputs (before weight)- important, see get_neuron_inputs function
        # TODO: add more types of neurons and example parameters
        # this was used only in tests (see article)
        if neuronModel == "iaf_psc_exp" or neuronModel == "iaf_psc_alpha":
            if params is not None:
                for name, val in params.items():
                    if name in IAF_PSC_EXP_PARAMS:
                        IAF_PSC_EXP_PARAMS[name] = val
                    else:
                        weights_init[name] = val

            nestNet.neuronsModel = neuronModel
            nestNet.model_params_dict = IAF_PSC_EXP_PARAMS
            nestNet.R = IAF_PSC_EXP_PARAMS['tau_m'] / IAF_PSC_EXP_PARAMS['C_m']
            nestNet.tau = IAF_PSC_EXP_PARAMS['tau_m']
            nestNet.x0 = IAF_PSC_EXP_PARAMS['V_reset']
            nestNet.t_ref = IAF_PSC_EXP_PARAMS['t_ref']
            nestNet.V_th = IAF_PSC_EXP_PARAMS['V_th']
            nestNet.tau_s = IAF_PSC_EXP_PARAMS['tau_syn_ex']
            if IAF_PSC_EXP_PARAMS['tau_syn_ex'] != IAF_PSC_EXP_PARAMS['tau_syn_in']:
                raise "Assumption: tau_s for exhibitory and inhibitory synapses should be the same"

            nestNet.refresh_nest()
            # find PSCs, and 1/max(I) - scaling factor
            # assumption is that t_s for exibitiory and inhibitory is the same
            # create 2 neurons
            n1 = nest.Create(neuronModel, params={'I_e': 1000000000.0})
            n2 = nest.Create(neuronModel)

            # connect 1 with 2 i 3 with weight 1 and -1 and no delay
            conn_spec_dict = {"rule": "one_to_one"}
            syn_spec_dict = {"weight": 1.0,
                             "model": nestNet.synapse_model,
                             "delay": nestNet.dt
                             }
            nest.Connect(n1, n2, conn_spec=conn_spec_dict, syn_spec=syn_spec_dict)

            # record postsynaptic neuron input exhibitory and inhibitory current
            multimetersParams = {"withtime": True,
                                 "record_from": ['I_syn_ex'],
                                 'interval': nestNet.dt}
            multimeter_ex = nest.Create('multimeter', 1, params=multimetersParams)
            nest.Connect(multimeter_ex, n2, "one_to_one", syn_spec=syn_spec_dict)

            # spikedetectorParams = {"withtime": True}
            # spikedetector = nest.Create('spike_detector', 1, params=spikedetectorParams)
            # nest.Connect(n1, spikedetector, "one_to_one", syn_spec=syn_spec_dict)

            # simulate
            nest.Simulate(nestNet.simTime + nestNet.dt)

            nestNet.PSC_ex = nest.GetStatus(multimeter_ex)[0]['events']['I_syn_ex'][1:]
            nestNet.PSC_in = -1 * nestNet.PSC_ex

            # check correctness:
            t = np.arange(0, 3*nestNet.model_params_dict['tau_syn_ex'], nestNet.dt)

            check_ex = t * np.exp(-t / nestNet.tau_s)
            nestNet.maxI_ex = max(check_ex)
            # check_ex = check_ex / nestNet.maxI_ex

            # check_in = t * np.exp(-t / nestNet.tau_s)
            # check_in = -1*check_in / max(check_in)

            # plt.figure()
            # plt.plot(nestNet.PSC_ex, label='target_ex')
            # plt.plot(check_ex[:len(nestNet.PSC_ex)], label='calc')
            # plt.legend()
            # plt.show()

            # plt.figure()
            # plt.plot(nestNet.PSC_in[:len(nestNet.PSC_ex)], label='target_in')
            # plt.plot(check_in[:len(nestNet.PSC_ex)], label='calc')
            # #plt.plot(-1*nestNet.PSC_ex[:len(nestNet.PSC_ex)], label="minus psc ex")
            # plt.legend()
            # plt.show()
            # print("here")
        else:
            raise """Not coded yet"""

    @staticmethod
    def refresh_nest():
        """
        Resets NEST's internal variables, recording times, neuron state variables, etc.
        Call this function for each new network (for each new response)!

        Arguments
        -------
            None.
        Returns
        -------
            None.

        """
        nest.ResetKernel()
        # set time step of simulation
        nest.SetKernelStatus({"resolution": nestNet.dt})
        # set spike times rounding to grid,
        # see NEST doc, if 'allow_offgrid_times == False,
        # offgrid spikes will be ignored
        nest.SetDefaults('spike_generator',   {'allow_offgrid_times': True})
        # set neuron params
        if not defaultModel:
            nest.SetDefaults(nestNet.neuronsModel, nestNet.model_params_dict)

    def inputs_to_inputs_times(self, inputs):
        """
        Scales linearly values from the range [1, 0] to the range
        of input spikes times [nestNet.minInSpikeTime, nestNet.maxInSpikeTime].
        Note that 1 corresponds to the nestNet.minInSpikeTime value and
        0 to the nestNet.maxInSpikeTime value.
        Does not validate inputs.

        Parameters
        ----------
        inputs : np.array
            Array of values from [0, 1] range.

        Returns
        -------
        np.array
            Array of values scaled to [nestNet.minInSpikeTime, nestNet.maxInSpikeTime].

        """
        return nestNet.maxInSpikeTime - inputs * nestNet.diffInSpikeTime

    def inputs_times_to_inputs(self, times):
        """
        Scales linearly values from the range
        [nestNet.minInSpikeTime, nestNet.maxInSpikeTime] to the [1, 0] range.
        Note that 1 corresponds to the nestNet.minInSpikeTime value and
        0 to the nestNet.maxInSpikeTime value.
        Does not validate inputs.

        Parameters
        ----------
        inputs : np.array
            Array of values from [nestNet.minInSpikeTime, nestNet.maxInSpikeTime] range.

        Returns
        -------
        np.array
            Array of values scaled to [1, 0].

        """
        return (nestNet.maxInSpikeTime - times) / nestNet.diffInSpikeTime

    def outputs_to_outputs_times(self, targets):
        """
        Scales values from the range [1, 0] to the range
        of input spikes times [nestNet.minOutSpikeTime, nestNet.maxOutSpikeTime].
        Note that 1 corresponds to the nestNet.minOutSpikeTime value and
        0 to the nestNet.maxOutSpikeTime value.
        Does not validate inputs.

        Parameters
        ----------
        targets : np.array
            Array of values from [0, 1] range.

        Returns
        -------
        np.array
            Array of values scaled to [nestNet.minOutSpikeTime, nestNet.maxOutSpikeTime].

        """
        return nestNet.maxOutSpikeTime - targets * nestNet.diffOutSpikeTime

    def outputs_times_to_outputs(self, times):
        """
        Linearly scales values from [nestNet.maxOutSpikeTime, nestNet.minOutSpikeTime] range
        to [0, 1]. Note that nestNet.maxOutSpikeTime correspods to 0 value, and
        nestNet.minOutSpikeTime to 1.

        Does not validate inputs.

        Parameters
        ----------
        times : np.array or list
            Array of times from [nestNet.maxOutSpikeTime, nestNet.minOutSpikeTime] range.

        Returns
        -------
        list
            List of scaled values. Values can be outside of range [0, 1].
        """
        return (nestNet.maxOutSpikeTime - times)/nestNet.diffOutSpikeTime

    def respond_open_loop(self, inputs):
        """
        Determines the open-loop response of the network -
        without feedbacks from the network's output to its inputs.

        Parameters
        ----------
        inputs : np.array or list
            Array of input data. Each row is a separate/next sample.

        Returns
        -------
        x: list
            List of net's responses. Each row is a respond on
            corresponding row from inputs.

        """
        # scale inputs to times
        inputs_spike_times = self.inputs_to_inputs_times(inputs)
        x = []
        # calc net's respond
        for in_k in inputs_spike_times:
            x.append(self.respond(in_k))
        return x

    def respond_close_loop(self, inputs, old_x):
        """
        Determines the close-loop response of the network -
        with feedbacks from the network's output to its inputs.

        Parameters
        ----------
        inputs : np.array or list
            Array of input data. Each row is a separate/next sample.
        x0 : np.array or list
            Initial state of net outputs.

        Returns
        -------
        x : list
            Array of net's responses. Each row is a respond on
            corresponding row from inputs.
        """
        x = []
        # scale inputs to times
        inputs_spike_times = self.inputs_to_inputs_times(inputs)
        for in_k in inputs_spike_times:
            # net inputs can not be out from [0, 1] range
            old_x = self.inputs_to_inputs_times(np.clip(old_x, 0.0, 1.0))
            # inputs in k instant are the real inputs in k instant and
            # one previous net respones
            inputs_k = np.concatenate((in_k, old_x))
            g_spikes_k = self.respond(inputs_k)
            # change output times to output values
            old_x = self.outputs_times_to_outputs(np.array(list(g_spikes_k.values())))
            # save answers
            x.append(old_x)
        return x

    def respond(self, inputs_spikes_times):
        """
        Generates a nest network response on one set of inputs.
        NEST network in only created here, in each time from scratch.

        Parameters
        ----------
        inputs_spikes_times : float np.array or list
            Input neuron firing times.

        Returns
        -------
        output_spikes : dict
            Dictionary of:
                output neuron number: its spike time or empty list if neuron has not fired.

        """
        # reset nest and set (neuron) model
        nestNet.refresh_nest()
        # change times to input spikes,
        # spikes are understand as pairs: neuron number: spike time
        inputs_spikes_times = np.append(inputs_spikes_times, nestNet.minInSpikeTime)
        # add addtitionaly spike in the middle of inputs spikes times
        middle_time = (nestNet.maxInSpikeTime-nestNet.minInSpikeTime)/2 + nestNet.minInSpikeTime
        inputs_spikes_times = np.append(inputs_spikes_times, middle_time)
        inputs_spikes = [{"spike_times": [x]} for x in inputs_spikes_times]
        """net inputs"""
        self.inputs_generators = nest.Create("spike_generator", n=self.hmIn,
                                             params=inputs_spikes)
        """neurons"""
        # output and hidden neurons
        self.neurons = nest.Create(nestNet.neuronsModel, n=self.hmOut+self.hmN)

        """connections/synapses"""
        syn_spec_dict = {"model": nestNet.synapse_model}
        for (preSyn, postSyn, d), weight in self.connections.items():
            syn_spec_dict["weight"] = [weight]
            syn_spec_dict["delay"] = d
            nest.Connect([preSyn], [postSyn], "one_to_one", syn_spec=syn_spec_dict)

        """measurements"""
        # what should be recorded
        what2rec = ['I_syn_in', 'I_syn_ex']  # 'V_m'
        multimeterParams = {"withtime": True,
                            "record_from": what2rec,
                            'interval': nestNet.dt}
        spikedetectorParams = {"withtime": True}

        # connect recorder divices to neurons
        syn_spec_dict['weight'] = 1.0
        # delay nestNet.dt because it is the minimum delay of connections between neurons
        syn_spec_dict['delay'] = nestNet.dt

        self.multimeter = nest.Create('multimeter', params=multimeterParams)
        nest.Connect(self.multimeter, self.neurons, "all_to_all", syn_spec=syn_spec_dict)

        self.spikedetector = nest.Create('spike_detector', params=spikedetectorParams)
        nest.Connect(self.inputs_generators+self.neurons,
                     self.spikedetector, 'all_to_all', syn_spec=syn_spec_dict)

        # simulate net for the given time in ms
        nest.Simulate(nestNet.simTime + nestNet.dt)

        return self.get_spikes(self.output_neurons)

    def train(self, inputs, targets, rate, hmEpochs=None, version='online', hm_workers=1,
              target_learning_MSE=None, target_test_MSE=None, target_MSE_spikes=None,
              test_inputs=None, test_targets=None,
              learning_MSE_gen=[], test_MSE_gen=[], learning_MSE_spikes_gen=[],
              task_id=None, dynamic=False, hmSamples=None,
              close_loop=False):
        """
        Network training. Operation is determined by the arguments.

        Parameters
        ----------
        inputs : float np.array
            Each row is a separate sample of inputs. The values should be in the range [0, 1].
        targets : float np.array
            Each row is a separate target response to the same row of input data.
            The values should be in the range [0, 1].
        rate : float, float list, float array
            Learning rate constant or specified for each learning epoch.
        hmEpochs : int, optional
            Maximal number of epochs. The default is None.
        version : string, optional
            "Online" or "batch". In online weights are changed after each sample. In batch after...
            batch of samples. Batch use parralel computation. See hm_workers.
            The default is 'online'.
        hm_workers : int, optional
            Number of parralel processes in batch learning. The default is 1.
        target_learning_MSE : float, optional
            If the network reaches this MSE value for the output values
            (those in the interval [0, 1]) on the learning data then the learning will end.
            The default is None.
        target_test_MSE : float, optional
            See target_learning_MSE, but here on verification data.  The default is None.
        target_MSE_spikes : float, optional
            See target_learning_MSE, but here on spikes times and learning data.
            The default is None.
        test_inputs : float np.array, optional
            Verification inputs. See inputs. The default is None.
        test_targets : float np.array, optional
            Verification targets. See targets. The default is None.
        learning_MSE_gen : list, optional
            MSE list for learning data in previous epochs, provide if you continue teaching.
            New MSE values will be added at the end. The default is [].
        test_MSE_gen : list, optional
            See learning_MSE_gen. Here MSE is calculated on verification data. The default is [].
        learning_MSE_spikes_gen : list, optional
            See learning_MSE_gen. Here MSE is calculated on spikes times on learning data.
            The default is [].
        x0 : float np.array, optional
            Initial state of net outputs. Use if recursive connection are used.
            The default is 0.
        task_id : int, optional
            State in the case of parallel online learning of multiple networks.
            DO NOT learn multiple networks parrallel in batch learning!
            The default is None.
        dynamic : bool, optional
            Define that net has recursive connections. The default is False.
        hmSamples : int, optional
            Set how many samples are to be used during each learning epoch.
            The samples will be selected randomly. The default is None, that means
            that all samples will be used in each epoch.

        Returns
        -------
        None.

        """
        self.version = version
        self.hm_workers = hm_workers
        self.dynamic = dynamic
        self.hmSamples = hmSamples
        self.close_loop = close_loop

        # check valid termination condition
        if hmEpochs is None and target_learning_MSE is None and target_test_MSE is None and\
           target_MSE_spikes is None:
            raise "Determine the stopping condition"
        if target_test_MSE is not None and (test_targets is None or test_inputs is None):
            raise "Test targets or test inputs is None, and stopping condition is test MSE."

        # check is rate constant
        if isinstance(rate, list) or isinstance(rate, np.ndarray):
            # if not
            const_rate_flag = False
            # check that the correct number of rates  is given
            if not hmEpochs or len(rate) != hmEpochs:
                raise 'The number of epochs and the given learning rate is not equal.'
        else:
            const_rate_flag = True
            rate_i = rate

        # if the network should have recursive connections from outputs to inputs
        if self.dynamic:
            if not self.close_loop:  # if open loop
                # in open loop inputs are [real inputs, measured targets]
                inputs = np.concatenate((inputs, targets), axis=1)
            self.x0 = targets[0, :]  # in values
            self.x0_times = self.inputs_to_inputs_times(self.x0)  # in input times
            # shift targets, inputs from sample k must give measured targets from k+1
            # estimation is in k on k+1
            targets = targets[1:]
            inputs = inputs[:targets.shape[0]]

        # calculate target spike times
        self.targets_spikes_times = self.outputs_to_outputs_times(targets)
        # calculate the spike times of the inputs
        self.inputs_spikes_times = self.inputs_to_inputs_times(inputs)

        # if use all samples in learning
        if self.hmSamples is None:
            self.hmSamples = self.targets_spikes_times.shape[0]

        # if they are test then calculate target spike times
        if test_inputs is not None:
            test_inputs_spikes_times = self.inputs_to_times(test_inputs)
        if test_targets is not None:
            test_targets_spikes_times = self.inputs_to_times(test_inputs)

        i = 0
        while not hmEpochs or i < hmEpochs:
            if not const_rate_flag:
                rate_i = rate[i]

            g, g_spikes_times, samples = self.train_epoch(rate_i)

            # TODO: USUŃ TO PLOTOWANIE
            # t = np.arange(0, learning_answer.shape[0], 1)
            # fig, ax = plt.subplots()
            # # target
            # plt.plot(t, targets_spikes_times[samples], 'b--', label='Reference trajectory')
            # # odpowiedź
            # plt.plot(answer_spikes_times, 'r', label='Net respond')
            # plt.legend()
            # plt.show()

            learning_MSE = self.mse(g, targets[samples])
            learning_MSE_gen.append(learning_MSE)

            learning_MSE_spikes = self.mse(g_spikes_times, self.targets_spikes_times[samples])
            learning_MSE_spikes_gen.append(learning_MSE_spikes)

            # kiedy warunek zakończenia jest od testowych jeśli są jakieś testowe
            if target_test_MSE is not None:
                # jezeli obiekt dynamiczny
                if dynamic:
                    print('sprawdz tutaj')
                    test_answer = self.respond_close_loop(test_inputs_spikes_times,
                                                          test_targets[0])
                else:  # jezeli obiekt jest statyczny
                    print('sprawdz tutaj')
                    test_answer = self.respond_open_loop(test_inputs_spikes_times)

                # licz mse
                test_MSE = self.mse(test_answer, test_targets)
                test_MSE_gen.append(test_MSE)

                print(
                    f"""{task_id}, {i}, train mse: {learning_MSE:.7f}, test mse {test_MSE:.7f}, spike_MSE {learning_MSE_spikes:.7f}""")
                # jezeli osiągnał target_test_MSE
                if target_test_MSE and test_MSE < target_test_MSE:
                    # to zakończ
                    print("Target test mse has been reached.")
                    break
            else:  # kiedy są tylko targetowe
                print(
                    f"""Task_id:{task_id}, Epoch:{i}, train mse: {learning_MSE:.7f}, spike_MSE {learning_MSE_spikes:.7f}""")

            # jezeli osiągnał target_learning_MSE
            if ((target_learning_MSE and learning_MSE < target_learning_MSE) or
                    (target_MSE_spikes and target_MSE_spikes > learning_MSE_spikes)):
                # to zakończ
                print("Target learning mse has been reached.")
                break
            i += 1

    def train_epoch(self, rate):
        """
        Train network through one epoch.

        Parameters
        ----------
        inputs_spikes_times : float np.array
            Times of inputs spikes. Each row is separated sample.
        targets_spikes_times : float np.array
            Target times of network output spikes. Each row is separetd target sample.
        rate : float
            Learning rate for this epoch.

        Returns
        -------
        g : list
            Net outputs in values. Each row is separated net output (row - one sample).
        g_spikes_times : float np.array
            Net outputs in spikes. (row - sample)
        samples : list
            List of indexes of used samples.
        """
        # draw idexes of samples used in learning epoch
        if self.close_loop:
            # in close loop samples must be in original order
            # so take some hmSamples samples in order
            max_start = self.targets_spikes_times.shape[0] - self.hmSamples
            idx_start = randint(0, max_start)
            idx_end = idx_start + self.hmSamples
            # indexex of samples used during learning epoch
            samples = range(idx_start, idx_end)

            if idx_start != 0:  # if samples do not start at 0
                # then x0 must be changed
                x0 = self.outputs_times_to_outputs(self.targets_spikes_times[idx_start-1])
                x0_times = self.inputs_to_inputs_times(x0)
            else:
                x0_times = self.x0_times
        else:
            # in open loop samples can be mixed up
            # draw some hmSamples samples
            samples = sample(range(0, self.targets_spikes_times.shape[0]), self.hmSamples)

        # take only drawn samples
        inputs_spikes_times = self.inputs_spikes_times[samples, :]
        targets_spikes_times = self.targets_spikes_times[samples, :]

        if self.version == 'online':
            # weights are changed in each sample, no parralel option
            # self.connections are changed inside train online
            if self.close_loop:  # ok
                g, g_spikes_times, not_fired,_ =\
                    self.train_online_close_loop(rate, inputs_spikes_times, targets_spikes_times,
                                                 x0_times)
            else:  # if open loop # ok
                g_spikes_times, not_fired,_=\
                    self.train_online_open_loop(rate, inputs_spikes_times, targets_spikes_times)

                g = self.outputs_times_to_outputs(g_spikes_times)

        elif self.version == 'batch':
            # batch - self.connections are changed only after batch of samples
            if not self.close_loop:  # in open loop  # ok
                pe = parallel_for_learning(self.hm_workers, self.train_batch_open_loop, self.close_loop)
                g_spikes_times, not_fired, conns =\
                    pe.evaluate(rate, inputs_spikes_times, targets_spikes_times)
                g = self.outputs_times_to_outputs(np.array(g_spikes_times))
            else:  # if close loop  # ok
                # divide inputs and targets in equally long batches
                part_inputs, part_targets = self.divide_data_equally(inputs_spikes_times,
                                                                     targets_spikes_times)
                pe = parallel_for_learning(self.hm_workers, self.train_batch_close_loop, self.close_loop)
                g, g_spikes_times, not_fired, conns =\
                    pe.evaluate(rate, part_inputs, part_targets, x0_times)

            # calc new weights from weight obtained in batches
            self.calc_and_save_mean_from_conns(conns)

        elif self.version == "online_in_minibatches":
            # online_in_minibatches - self.connections are changed after each sample in minibatch,
            # and then finnaly after all minibatches as mean of weights from each minibatch

            # divide inputs and targets in equally long batches
            part_inputs, part_targets = self.divide_data_equally(inputs_spikes_times,
                                                                 targets_spikes_times)
            if not self.close_loop:  # in open loop
                pe = parallel_for_learning(self.hm_workers, self.train_online_open_loop, self.close_loop)
                g_spikes_times, not_fired, conns =\
                    pe.evaluate(rate, part_inputs, part_targets)
                g = self.outputs_times_to_outputs(np.array(g_spikes_times))
            else:  # if close loop

                pe = parallel_for_learning(self.hm_workers, self.train_online_close_loop,
                                           self.close_loop)
                g, g_spikes_times, not_fired, conns =\
                    pe.evaluate(rate, part_inputs, part_targets, x0_times)

            # calc new weights from weight obtained in batches
            self.calc_and_save_mean_from_conns(conns)

        self.find_nonspiking_on_all_samples(not_fired)
        # zwiekszanie wag dla neuronów które nie wystrzeliły ani razu
        # self.increase_weight_of_nonfiring(self.not_fired_at_all)
        return g, g_spikes_times, samples

    def train_batch_open_loop(self, rate, inputs_spikes_times_k, targets_spikes_times_k):
        # here function allways returns only one samples of g_spike_k
        g_spikes_k = self.respond(inputs_spikes_times_k)

        conn_with_new_weights, not_fired_hidden =\
            self.calc_new_weights(rate, g_spikes_k, targets_spikes_times_k)
        return list(g_spikes_k.values()), not_fired_hidden, conn_with_new_weights

    def train_online_open_loop(self, rate, inputs_spikes_times, targets_spikes_times):
        not_fired = []  # list on neurons, that did not fired on any sample
        g_spikes_times = []
        for inputs_k, targets_k in zip(inputs_spikes_times, targets_spikes_times):
            # feedforward
            g_spikes_k = self.respond(inputs_k)
            g_spikes_times.append(list(g_spikes_k.values()))
            # weights change
            self.connections, not_fired_hidden =\
                self.calc_new_weights(rate, g_spikes_k, targets_k)
            not_fired.append(not_fired_hidden)
        return np.stack(g_spikes_times), not_fired, self.connections

    def train_batch_close_loop(self, rate, inputs_spikes_times, targets_spikes_times, old_x_times):
        not_fired = []  # list on neurons, that did not fired on any sample
        conn_with_new_weights = []
        g = []
        g_spikes_times = []
        for in_k, target_k in zip(inputs_spikes_times, targets_spikes_times):
            not_fired_hidden = []
            # inputs in k instant are the real inputs and previous net respones
            inputs_k = np.concatenate((in_k, old_x_times))
            # calc net respond
            g_spikes_k = self.respond(inputs_k)
            # save outputs spikes times
            foo = list(g_spikes_k.values())
            g_spikes_times.append(foo)
            # change output times to output values
            g.append(self.outputs_times_to_outputs(np.array(foo)))

            # weights change
            conn, not_fired_hidden =\
                self.calc_new_weights(rate, g_spikes_k, target_k)
            conn_with_new_weights.append(conn)

            # recursive singals from outputs to inputs:
            # change old_x values to input spike times
            old_x_times = self.inputs_to_inputs_times(np.clip(g[-1], 0.0, 1.0))

            not_fired.append(not_fired_hidden)

        return np.stack(g), np.stack(g_spikes_times), not_fired, conn_with_new_weights

    def train_online_close_loop(self, rate, inputs_spikes_times, targets_spikes_times, old_x_times):
        not_fired = []  # list on neurons, that did not fired on any sample
        g = []
        g_spikes_times = []
        for in_k, target_k in zip(inputs_spikes_times, targets_spikes_times):
            not_fired_hidden = []
            # inputs in k instant are the real inputs and previous net respones
            inputs_k = np.concatenate((in_k, old_x_times))
            # calc net respond
            g_spikes_k = self.respond(inputs_k)
            # save outputs spikes times
            foo = list(g_spikes_k.values())
            g_spikes_times.append(foo)
            # change output times to output values and save them
            g.append(self.outputs_times_to_outputs(np.array(foo)))

            # weights change
            self.connections, not_fired_hidden =\
                self.calc_new_weights(rate, g_spikes_k, target_k)

            # recursive singals from outputs to inputs:
            # change old_x values to input spike times
            old_x_times = self.inputs_to_inputs_times(np.clip(g[-1], 0.0, 1.0))

            not_fired.append(not_fired_hidden)

        return np.stack(g), np.stack(g_spikes_times), not_fired, [self.connections]

    def calc_and_save_mean_from_conns(self, conns):
        # diviade all new weights by key
        weights = {k:[w] for k, w in conns[0].items()}
        for conn in conns[1:]:
            for key, w in conn.items():
                # if w == self.connections[key]: # weź tylko te wagi ktore sie zmieniły
                #     continue
                weights[key].append(w)
        # get mean of new weights
        for key, w in weights.items():
            self.connections[key] = np.mean(weights[key])

    def find_nonspiking_on_all_samples(self, not_fired):
        # weź nie spikajace z pierwszej próbki
        self.not_fired_at_all = not_fired[0]
        # dla każdych nie spikajacych z dalszych próbek
        for l in not_fired[1:]:
            # przepisz tylko te które jeszcze nie spikneły
            self.not_fired_at_all = [x for x in l if x in self.not_fired_at_all]
            # jeżeli lista tych których nie spikneły ani razu jest pusta
            # to znaczy że każdy przynajmniej raz już spiknął
            if not self.not_fired_at_all:
                break

    def calc_new_weights(self, rate, answer_spikes, targets_spikes_times):
        not_fired_hidden = []
        # change targets values to targets spikes
        targets_spikes = dict(zip(self.output_neurons, targets_spikes_times))

        # find input currents and spikes of hidden and output neurons,
        # will be useful for learning
        self.neurons_inputs = self.get_neurons_input_current()

        self.spikes = self.get_spikes(self.input_neurons
                                      + self.hidden_neurons) | answer_spikes

        # count d for output neurons
        d_output = self.calc_output_d(targets_spikes, answer_spikes)

        # get the new weights of output neurons
        conn_with_new_weights = self.update_output_weights(d_output, rate, answer_spikes)

        # get the new weights of hiddem neurons
        conn_with_new_weights.update(
            self.update_hidden_weights(d_output, rate, not_fired_hidden,
                                       layers=self.layers))
        return conn_with_new_weights, not_fired_hidden

    def increase_weight_of_nonfiring(self, not_fired):
        # funkcja dziala tylko dla ukrytych
        for i_idx in not_fired:
            pass
            # self.V[i_idx] += 0.005

    def update_output_weights(self, d, rate, output_spikes):
        conn_with_new_weights = {}
        for nID in self.output_neurons:
            n_spike_time = output_spikes[nID]
            # find connections to nID output neuron
            conns = self.find_connections(to_node=nID)
            for (preSyn, postSyn, delay), weight in conns.items():
                # take the delayed input current (before weight) only from this connection,
                i = self.get_current_before_weight(preSyn, delay)
                # calculate 3) old_s_time, s_time, I
                dxdw = self.calc_dxdw(0, self.spikes[postSyn], i)
                # # update zmiany wagi
                old_w = weight
                conn_with_new_weights[preSyn, postSyn, delay] = weight - rate * d[postSyn] * dxdw
                #conn_with_new_weights[preSyn, postSyn, delay] = old_w

                if weights_only_positive and conn_with_new_weights[preSyn, postSyn, delay] < 0:
                    conn_with_new_weights[preSyn, postSyn, delay] = old_w

        return conn_with_new_weights

    def calc_output_d(self, target_spikes, output_spikes):
        """
        Calculated of ,,d" for outputs neurons.

        Parameters
        ----------
        target_spikes : dict
            Dictionary of: neuornsID: time of its first spike
        output_spikes : dict
            Dictionary of: neuornsID: time of target spike time

        Returns
        -------
        d : dict
            d - Errors of output neurons. Dictionary of: neuronsID: its d.
        """
        d = {}
        # for each output neuron
        for nID in self.output_neurons:
            # calc equation 1
            dEdt = self.calc_output_dEdt(target_spikes[nID], output_spikes[nID])
            # calc 2 for output,
            # function to calc 2 is the same for all neurons
            dtdx = self.calc_dtdx(nID)
            d[nID] = dEdt * dtdx
        return d

    def calc_output_dEdt(self, target_time, output_time):
        """
        Calculates dEdt of output nerons.
        The dEdt is the difference between the current first spike time and
        the target time of the neuron first spike.

        Parameters
        ----------
        target_time : float
            Target time of neuron spike.
        output_time : float
            Current time of neuron spike.

        Returns
        -------
        float
            dEdt of output neuron.

        """
        return output_time - target_time

    def calc_dtdx(self, nID):
        """
        Calculate dtdx of each neuron

        Parameters
        ----------
        nID : int
            Neuron ID.

        Returns
        -------
        denominator : float
            Calcluated value of denominator.

        """
        # find inputs current to the neuron
        Iin = self.neurons_inputs[nID]
        # choose one way of calc this:
        # 1 way:
        # denominator = self.calc_dtdx_denominator(0, self.spikes[nID], Iin)
        # 2 way, done simpler:
        idx = round(self.spikes[nID] / nestNet.dt)
        try:
            denominator = (nestNet.V_th - nestNet.R * Iin[idx]) / nestNet.tau
        except IndexError:
            denominator = (nestNet.V_th - nestNet.R * Iin[-1]) / nestNet.tau

        # The denominator can be zero when no non zero inputs to the neuron
        if abs(denominator) < 0.1:
            # print(f'Limit of denominator, value {denominator}.')
            if denominator >= 0:
                denominator = 0.1
            else:
                denominator = -0.1
        return 1/denominator

    def calc_dtdx_denominator(self, old_s_time, s_time, i):
        """
        Calculate denominator of dtdx equation.

        Parameters
        ----------
        old_s_time : float
            Time of prevoius spike. In the case of first spike or
            when only one spike per simulaton is possible enter ,,0".
        s_time : float
            Current spike time.
        i : float np.array
            Input current to neuron.

        Returns
        -------
        float
            Value of denominator.

        """
        integral, i_in_spike_time = self.calc_integral_from_eI(old_s_time, s_time, i)
        return math.exp(-s_time/nestNet.tau) / nestNet.tau * \
            (math.exp(old_s_time/nestNet.tau) * nestNet.x0 +
             nestNet.R * integral / nestNet.tau) -\
            nestNet.R * i_in_spike_time / nestNet.tau

    def calc_integral_from_eI(self, old_s_time, s_time, I):
        """
        Calculates the integral from the expression e^(t/t_m)*I(t) dt
        If neuron has not fired assume that its fire at nestNet.simTime
        Parameters
        ----------
        old_s_time : float
            Time of previous spike, so far always 0.
            (due to only one spike per neuron assumption).
        s_time : float
            Time of spike.
        I : float np.array
            Input current to neuron.

        Returns
        -------
        float
            Calculated integral.
        float
            Value of input current in spike time.

        """
        i = I[round(old_s_time/nestNet.dt):round(s_time/nestNet.dt)+1]
        t = np.linspace(0, s_time, round((s_time-old_s_time)/nestNet.dt)+1)
        return np.trapz(np.exp(t/nestNet.tau) * i, t), i[-1]

    def update_hidden_weights(self, neuron_errors, rate, not_fired, layers=None):
        # divide the network into layers
        if layers is None:
            # reject delays from connections, take a call only once
            conn = []
            for preSyn, postSyn, d in self.connections.keys():
                if (preSyn, postSyn) not in conn:
                    conn.append((preSyn, postSyn))
            layers = graphs.anyNet_layers(self.input_neurons, self.output_neurons, conn)

        conn_with_new_weights = {}
        # for each layer from the back, the error propagates from outputs to inputs
        for lay in reversed(layers):
            # find neurons hidden in the layer (not input and not output)
            lay_hidden_neurons = [n for n in lay if n in self.hidden_neurons]
            # for each neuron hidden in the layer
            for nID in lay_hidden_neurons:
                # find connections to nID hidden neuron
                conn = self.find_connections(to_node=nID)
                # if the hidden neuron did not fire at all
                # rewrite its weights
                # TODO consider artificial increase of weights
                if self.spikes[nID] >= nestNet.simTime:
                    for key, w in conn.items():
                        conn_with_new_weights[key] = w
                    not_fired.append(nID)
                    continue
                # calc 1)  dEdt
                dedt = self.calc_hidden_dEdt(nID, neuron_errors)
                # calc 2) dtdx
                dtdx = self.calc_dtdx(nID)
                # save d of neuron
                neuron_errors[nID] = dedt * dtdx
                # for each connections to nID neuron, that is for each weight of nID neuron
                for (preSyn, postSyn, delay), weight in conn.items():
                    # take the delayed input current (before weight) only from this connection,
                    # this in input current to postSyn neuron, that is
                    # it's output of preSyn neuron
                    i = self.get_current_before_weight(preSyn, delay)
                    # licz 3) dxdw,  old_s_time, s_time, I
                    dxdw = self.calc_dxdw(0, self.spikes[postSyn], i)
                    # change weight
                    old_w = weight
                    conn_with_new_weights[(preSyn, postSyn, delay)] =\
                        weight - rate * neuron_errors[postSyn] * dxdw

                    if weights_only_positive and conn_with_new_weights[(preSyn, postSyn, delay)] < 0:
                        # to waga równa jest zero
                        conn_with_new_weights[(preSyn, postSyn, delay)] = old_w
        return conn_with_new_weights

    def calc_hidden_dEdt(self, preSynID, neuron_errors):
        """
        Calculates  dEdt for hidden neurons.
        dEdt is dependent on postsynaptic errors relative to the preSynID neuron.

        Parameters
        ----------
        preSynID : int,
            Neuron ID whose dedt is calculated.
        neuron_errors : dict
            Pairs of neuron ID: neuron error (delta).

        Returns
        -------
        dEdt : float
            Value of dEdt.

        """

        dEdt = 0
        t_i = self.spikes[preSynID]
        # find outgoing calls from neuron nID
        conn = self.find_connections(from_node=preSynID)
        # for each outgoing call from nID:   for j in J and k in K
        # preSyn - neuron i; postSyn - neuron j
        tau_m = nestNet.tau
        tau_s = nestNet.tau_s
        for (preSyn, postSyn, delay), weight in conn.items():
            t_j = self.spikes[postSyn]
            # if the current of this synapse did not affect the firing of the postSyn neuron
            if t_i+delay > t_j:  # if not t_j >= t_i+delay
                continue

            foo = tau_m-tau_s
            limit = 0.1
            if abs(foo) < limit:
                if foo >= 0.0:
                    foo = limit
                else:
                    foo = -limit
            a = tau_m / (nestNet.maxI_ex * foo**2)
            d = tau_m / (nestNet.maxI_ex * foo)
            # a = tau_m / (nestNet.maxI_ex * (tau_m-tau_s)**2)
            # d = tau_m / (nestNet.maxI_ex*(tau_m-tau_s))

            b = (t_j*tau_s - tau_m*t_j - tau_m*tau_s) * np.exp(t_j/tau_m + (t_i+delay-t_j)/tau_s)
            c = (tau_m*delay + tau_m*t_i + tau_m*tau_s - tau_s*delay - tau_s*t_i) *\
                np.exp((delay+t_i)/tau_m)
            part1 = a*(b+c)

            e = delay + t_i + tau_s
            f = np.exp((delay + t_i) / tau_m) - np.exp(t_j/tau_m + (delay+t_i-t_j)/tau_s)
            part2 = d*e*f

            ddt = part1 - part2

            dudt = weight * nestNet.R * np.exp(-t_j/tau_m) * ddt / tau_m

            dEdt += neuron_errors[postSyn] * dudt
        return dEdt

    def calc_dxdw(self, old_s_time, s_time, I):
        integral, _ = self.calc_integral_from_eI(old_s_time, s_time, I)
        return nestNet.R * math.exp(-s_time/nestNet.tau) * integral / nestNet.tau

    """###############################################################################
        end of backpropagation functions
    """

    def delay_signal(self, signal, delay):
        hm_d = round(delay / nestNet.dt)
        return np.concatenate(([0]*hm_d, signal))

    def get_current_before_weight(self, preSynID, delay=0):
        """
        Returns the output of the preSynID neuron transmitted across the synapse with a time delay
        in time from the beginning of the simulation to the end.
        The returned signal is correspondingly delayed, but it is from before the weight.
        Works on the assumption that the neuron can fire once.

        Parameters
        ----------
        preSynID : int
            ID of source (presynaptic) neuron.
        delay : float
            Synapitc delay.

        Returns
        -------
        i: float np.array
            Synapse current.
        """
        i = self.delay_signal(nestNet.PSC_ex, delay + self.spikes[preSynID])
        return i[0:nestNet.simulationSamples+1]

    def divide_signal_by_senders(self, signal, senders):
        """
        Function divides the signal into signals
        sent from each source/sender.

        Parameters
        ----------
        signal : float np.array
            Array of samples from diffrent senders.
        senders : int np.array
            Array of IDs of senders.

        Returns
        -------
        r : dict
            Dictionary of: senderID: signal send by this sender.
        """

        r = {}
        for sample, sender in zip(signal, senders):
            if sender in r:
                r[sender].append(sample)
            else:
                r[sender] = [sample]
        return r

    def find_samples_to_neuron(self, nID, signal, senders):
        """
        Finds samples from the signal sended only
        by neuron with the given nID.

        Parameters
        ----------
        nID : int
            Neuron ID.
        signal : np.array or list
            Signal from which samples will be sought.
        senders : np.array or list of ints
            ID's of the individuals sending a given sample in the signal.

        Returns
        -------
        r : list
            Signal sended by neuron with nID.

        """
        r = []
        for sample, sender in zip(signal, senders):
            if sender == nID:
                r.append(sample)
        return r

    def get_neurons_input_current(self):
        """
        Returns the input currents of all hidden and output neurons.

        Returns
        -------
        I : dict
            Dictionary of: neuronID: its input current.

        """
        foo = nest.GetStatus(self.multimeter)[0]['events']
        # sum of excitatory and inhibitory currents of ALL neurons
        I = foo['I_syn_in'] + foo['I_syn_ex']

        # divide samples by senders
        I = self.divide_signal_by_senders(I, foo['senders'])

        # The simulator starts recording from time t=nestNet.dt
        # and I want signals from t=0
        # so I extend the input current by 0 on the start
        I = {k: np.concatenate(([0], i)) for k, i in I.items()}
        # return self.delay_signal(inputs, delay)
        return I

    # def get_neurons_voltages(self):
    #     """
    #     Returns trajectories of voltages of all hidden and output neurons.

    #     Returns
    #     -------
    #     neurons_volateges : dict
    #         neuronID: samples of volategs in float np.array.

    #     """
    #     foo = nest.GetStatus(self.multimeter)[0]['events']
    #     # divide samples by senders
    #     neurons_voltages = self.divide_signal_by_senders(foo['V_m'], foo['senders'])

    #     # The simulator starts recording from time t=nestNet.dt
    #     # and I want signals from t=0
    #     # so I extend the input current by 0 on the start
    #     neurons_voltages = {k: np.concatenate(([0], u)) for k, u in neurons_voltages.items()}
    #     # return self.delay_signal(inputs, delay)
    #     return neurons_voltages

    def get_spikes(self, neuronsID):
        """
        The function returns all the spikes of the selected neurons.
        Returned dict is sorted in ascending order of neuronsID.
        The function does not validate the input data.

        If neuron didn't fire return nestNet.simTime!

        Parameters
        ----------
        neuronsID : list, touple, set
            Id of neurons which spikes are to be returned.

        Returns
        -------
        spikes : dict
            Dictionary of pairs: neurons IDs: all spikes times.
            If neuron didn't fire return nestNet.simTime!
        """

        foo = nest.GetStatus(self.spikedetector)[0]['events']
        spikes = {n: nestNet.simTime for n in sorted(neuronsID)}
        for time, sender in zip(foo['times'], foo['senders']):
            # save spikes only from selected neurons
            if sender in spikes:
                spikes[sender] = time
        # write nestNet.simTime for not fired neurons
        return spikes

    def find_connections(self, to_node=None, from_node=None, operator="or"):
        """
        Deppending on arguments finds connections to or from (or both) given neurons.

        Parameters
        ----------
        to_node : list or int, optional
            IDs of target neurons. The default is None.
        from_node : list or int, optional
            IDs of source neurons. The default is None.
        operator : string, optional
            ,,or" or ,,and". Defines the logical function between source and target neuron IDs.
            The default is "or".

        Returns
        -------
        r : dict
            Return dictionary of connections: connection: weight.

        """
        if to_node and not from_node:
            # if find connections only to node
            def condition(conn): return conn[1] == to_node
        elif not to_node and from_node:
            # if find connections only from node
            def condition(conn): return conn[0] == from_node
        else:
            # if both: to and from
            if operator == "or":
                def condition(conn): return conn[0] == from_node or conn[1] == to_node
            elif operator == "and":
                def condition(conn): return conn[0] == from_node and conn[1] == to_node
        r = {conn: weight for conn, weight in self.connections.items() if condition(conn)}
        return r

    def divide_data_equally(self, inputs, targets):
        # divide samples equally
        part_len = math.ceil(self.hmSamples / self.hm_workers)
        part_inputs = []
        part_targets = []
        for i in range(self.hm_workers):
            part_start_idx = i*part_len
            if part_start_idx >= inputs.shape[0]:
                break
            part_end_idx = (i+1)*part_len
            part_inputs.append(inputs[part_start_idx:part_end_idx, :])
            part_targets.append(targets[part_start_idx:part_end_idx, :])
        return part_inputs, part_targets

    def mse(self, x, target):
        return ((x - target)**2).mean()
