import numpy as np
from multiprocessing import Pool

class parallel_run(object):
    def __init__(self, num_workers, function, timeout=None):
        self.num_workers = num_workers
        self.function = function
        self.timeout = timeout
        self.pool = Pool(num_workers)

    def __del__(self):
        self.pool.close()  # should this be terminate?
        self.pool.join()

    def evaluate(self):
        jobs = []
        i = 0
        # for each part
        for i in range(self.num_workers):
            # wywołaj funkcję
            # self.function(i)
            jobs.append(self.pool.apply_async(self.function, (i,)))
            i = i+1
        # assign the returns
        save = []
        for job in jobs:
            save.append(job.get(timeout=self.timeout))
        return save


class parallel_for_learning(object):
    def __init__(self, num_workers, function, close_loop, timeout=None):
        self.num_workers = num_workers
        self.function = function
        self.timeout = timeout
        self.pool = Pool(num_workers)
        self.close_loop = close_loop

    def __del__(self):
        self.pool.close()  # should this be terminate?
        self.pool.join()

    def evaluate(self, rate, inputs_spikes_times, targets_spikes_times, x0_times=None):
        # specially done in such ugly way to gain minimal time difference,
        # close_loop is checked only once
        jobs = []
        if self.close_loop:
            # for each sample or batch
            for inputs, targets in zip(inputs_spikes_times, targets_spikes_times):
                # wywołaj funkcję
                # self.function(rate, inputs, targets, x0_times)
                jobs.append(self.pool.apply_async(self.function, (rate, inputs, targets, x0_times)))
                x0_times = targets[-1]
            return self.assign_returns_close_loop(jobs)
        else:  # open loop
            # for each sample or batch
            for inputs, targets in zip(inputs_spikes_times, targets_spikes_times):
                # wywołaj funkcję
                # g_spikes_times, not_fired_hidden, conn_with_new_weights =\
                # self.function(rate, inputs, targets)
                jobs.append(self.pool.apply_async(self.function, (rate, inputs, targets)))
            return self.assign_returns_open_loop(jobs)

    def assign_returns_open_loop(self, jobs):
        # assign the returns
        conns = []
        not_fired = []
        g_spikes_times = []
        for job in jobs:
            part_g_spikes_times, part_not_fired, part_conns = job.get(timeout=self.timeout)
            conns.append(part_conns)  # conns = conns + part_conns
            not_fired.append(part_not_fired)
            g_spikes_times.append(part_g_spikes_times)
        if len(not_fired[0]) > 1:
            not_fired = [f for l in not_fired for f in l]
        return np.vstack(g_spikes_times), not_fired, conns

    def assign_returns_close_loop(self, jobs):
        conns = []
        not_fired = []
        g_spikes_times = []
        g = []
        for job in jobs:
            part_g, part_g_spikes_times, part_not_fired, part_conns = job.get(timeout=self.timeout)
            conns = conns + part_conns
            not_fired = not_fired + part_not_fired
            g_spikes_times.append(part_g_spikes_times)
            g.append(part_g)
        return np.vstack(g), np.vstack(g_spikes_times), not_fired, conns
