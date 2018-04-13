import time


class TimeLoop:
    def __init__(self):
        self._preRunFunctions = []
        self._postRunFunctions = []
        self._timeStepFunctions = []
        self._functionNames = []
        self.time_steps_run = 0

    def add_step(self, step_obj):
        if hasattr(step_obj, 'pre_run'):
            self.add_pre_run_function(step_obj.pre_run)
        if hasattr(step_obj, 'post_run'):
            self.add_post_run_function(step_obj.post_run)
        self.add(step_obj.time_step, step_obj.name)

    def add(self, time_step_function, name=None):
        if name is None:
            name = str(time_step_function)
        self._timeStepFunctions.append(time_step_function)
        self._functionNames.append(name)

    def add_kernel(self, data_handling, kernel_func, name=None):
        self.add(lambda: data_handling.run_kernel(kernel_func), name)

    def add_pre_run_function(self, f):
        self._preRunFunctions.append(f)

    def add_post_run_function(self, f):
        self._postRunFunctions.append(f)

    def run(self, time_steps=1):
        self.pre_run()

        try:
            for i in range(time_steps):
                self.time_step()
        except KeyboardInterrupt:
            pass

        self.post_run()

    def benchmark_run(self, time_steps=0, init_time_steps=0):
        self.pre_run()
        for i in range(init_time_steps):
            self.time_step()

        start = time.perf_counter()
        for i in range(time_steps):
            self.time_step()
        end = time.perf_counter()
        self.post_run()

        time_for_one_iteration = (end - start) / time_steps
        return time_for_one_iteration

    def benchmark(self, time_for_benchmark=5, init_time_steps=10, number_of_time_steps_for_estimation=20):
        """
        Returns the time in seconds for one time step

        :param time_for_benchmark: number of seconds benchmark should take
        :param init_time_steps: number of time steps run initially for warm up, to get arrays into cache etc
        :param number_of_time_steps_for_estimation: time steps run before real benchmarks, to determine number of time
                                                    steps that approximately take 'time_for_benchmark'
        """
        # Run a few time step to get first estimate
        duration_of_time_step = self.benchmark_run(number_of_time_steps_for_estimation, init_time_steps)

        # Run for approximately 'time_for_benchmark' seconds
        time_steps = int(time_for_benchmark / duration_of_time_step)
        time_steps = max(time_steps, 20)
        return self.benchmark_run(time_steps, init_time_steps)

    def pre_run(self):
        for f in self._preRunFunctions:
            f()

    def post_run(self):
        for f in self._postRunFunctions:
            f()

    def time_step(self):
        for f in self._timeStepFunctions:
            f()
        self.time_steps_run += 1
