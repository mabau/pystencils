import time

from pystencils.integer_functions import modulo_ceil


class TimeLoop:
    def __init__(self, steps=2):
        self._call_data = []
        self._fixed_steps = steps
        self._pre_run_functions = []
        self._post_run_functions = []
        self._single_step_functions = []
        self.time_steps_run = 0

    @property
    def fixed_steps(self):
        return self._fixed_steps

    def add_pre_run_function(self, f):
        self._pre_run_functions.append(f)

    def add_post_run_function(self, f):
        self._post_run_functions.append(f)

    def add_single_step_function(self, f):
        self._single_step_functions.append(f)

    def add_call(self, functor, argument_list):
        if hasattr(functor, 'kernel'):
            functor = functor.kernel
        if not isinstance(argument_list, list):
            argument_list = [argument_list]

        for argument_dict in argument_list:
            self._call_data.append((functor, argument_dict))

    def pre_run(self):
        for f in self._pre_run_functions:
            f()

    def post_run(self):
        for f in self._post_run_functions:
            f()

    def run(self, time_steps=1):
        self.pre_run()
        fixed_steps = self._fixed_steps
        call_data = self._call_data
        main_iterations, rest_iterations = divmod(time_steps, fixed_steps)
        try:
            for _ in range(main_iterations):
                for func, kwargs in call_data:
                    func(**kwargs)
                self.time_steps_run += fixed_steps
            for _ in range(rest_iterations):
                for func in self._single_step_functions:
                    func()
                self.time_steps_run += 1
        except KeyboardInterrupt:
            pass
        self.post_run()

    def benchmark_run(self, time_steps=0, init_time_steps=0):
        init_time_steps_rounded = modulo_ceil(init_time_steps, self._fixed_steps)
        time_steps_rounded = modulo_ceil(time_steps, self._fixed_steps)
        call_data = self._call_data

        self.pre_run()
        for i in range(init_time_steps_rounded // self._fixed_steps):
            for func, kwargs in call_data:
                func(**kwargs)
        self.time_steps_run += init_time_steps_rounded

        start = time.perf_counter()
        for i in range(time_steps_rounded // self._fixed_steps):
            for func, kwargs in call_data:
                func(**kwargs)
        end = time.perf_counter()
        self.time_steps_run += time_steps_rounded
        self.post_run()

        time_for_one_iteration = (end - start) / time_steps
        return time_for_one_iteration

    def run_time_span(self, seconds):
        iterations = 0
        self.pre_run()
        start = time.perf_counter()
        while time.perf_counter() < start + seconds:
            for func, kwargs in self._call_data:
                func(**kwargs)
            iterations += self._fixed_steps
        end = time.perf_counter()
        self.post_run()
        self.time_steps_run += iterations
        return iterations, end - start

    def benchmark(self, time_for_benchmark=5, init_time_steps=2, number_of_time_steps_for_estimation='auto'):
        """Returns the time in seconds for one time step.

        Args:
            time_for_benchmark: number of seconds benchmark should take
            init_time_steps: number of time steps run initially for warm up, to get arrays into cache etc
            number_of_time_steps_for_estimation: time steps run before real benchmarks, to determine number of time
                                                 steps that approximately take 'time_for_benchmark' or 'auto'
        """
        # Run a few time step to get first estimate
        if number_of_time_steps_for_estimation == 'auto':
            self.run(1)
            iterations, total_time = self.run_time_span(0.5)
            duration_of_time_step = total_time / iterations
        else:
            duration_of_time_step = self.benchmark_run(number_of_time_steps_for_estimation, init_time_steps)

        # Run for approximately 'time_for_benchmark' seconds
        time_steps = int(time_for_benchmark / duration_of_time_step)
        time_steps = max(time_steps, 4)
        return self.benchmark_run(time_steps, init_time_steps)
