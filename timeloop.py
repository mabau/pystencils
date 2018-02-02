import time


class TimeLoop:
    def __init__(self):
        self._preRunFunctions = []
        self._postRunFunctions = []
        self._timeStepFunctions = []
        self.timeStepsRun = 0

    def addStep(self, stepObj):
        if hasattr(stepObj, 'preRun'):
            self.addPreRunFunction(stepObj.preRun)
        if hasattr(stepObj, 'postRun'):
            self.addPostRunFunction(stepObj.postRun)
        self.add(stepObj.timeStep)

    def add(self, timeStepFunction):
        self._timeStepFunctions.append(timeStepFunction)

    def addKernel(self, dataHandling, kernelFunc):
        self._timeStepFunctions.append(lambda: dataHandling.runKernel(kernelFunc))

    def addPreRunFunction(self, f):
        self._preRunFunctions.append(f)

    def addPostRunFunction(self, f):
        self._postRunFunctions.append(f)

    def run(self, timeSteps=0):
        self.preRun()

        try:
            for i in range(timeSteps):
                self.timeStep()
        except KeyboardInterrupt:
            pass

        self.postRun()

    def benchmarkRun(self, timeSteps=0, initTimeSteps=0):
        self.preRun()
        for i in range(initTimeSteps):
            self.timeStep()

        start = time.perf_counter()
        for i in range(timeSteps):
            self.timeStep()
        end = time.perf_counter()
        self.postRun()

        timeForOneIteration = (end - start) / timeSteps
        return timeForOneIteration

    def benchmark(self, timeForBenchmark=5, initTimeSteps=10, numberOfTimeStepsForEstimation=20):
        """
        Returns the time in seconds for one time step

        :param timeForBenchmark: number of seconds benchmark should take
        :param initTimeSteps: number of time steps run initially for warm up, to get arrays into cache etc
        :param numberOfTimeStepsForEstimation: time steps run before real benchmarks, to determine number of time steps
                                               that approximately take 'timeForBenchmark'
        """
        # Run a few time step to get first estimate
        durationOfTimeStep = self.benchmarkRun(numberOfTimeStepsForEstimation, initTimeSteps)

        # Run for approximately 'timeForBenchmark' seconds
        timeSteps = int(timeForBenchmark / durationOfTimeStep)
        timeSteps = max(timeSteps, 20)
        return self.benchmarkRun(timeSteps, initTimeSteps)

    def preRun(self):
        for f in self._preRunFunctions:
            f()

    def postRun(self):
        for f in self._postRunFunctions:
            f()

    def timeStep(self):
        for f in self._timeStepFunctions:
            f()
        self.timeStepsRun += 1




