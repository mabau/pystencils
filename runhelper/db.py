import time
import socket
from collections import OrderedDict

import blitzdb
import pandas as pd
from pystencils.cpu.cpujit import getCompilerConfig


def removeConstantColumns(df):
    remainingDf = df.loc[:, (df != df.ix[0]).any()]
    constants = df.loc[:, (df == df.ix[0]).all()].ix[0]
    return remainingDf, constants


class Database(object):
    class SimulationResult(blitzdb.Document):
        pass

    def __init__(self, file):
        self.backend = blitzdb.FileBackend(file)
        self.backend.autocommit = True

    @staticmethod
    def getEnv():
        return {
            'timestamp': time.mktime(time.gmtime()),
            'hostname': socket.gethostname(),
            'cpuCompilerConfig': getCompilerConfig(),
        }

    def save(self, params, result, env=None):
        documentDict = {
            'params': params,
            'result': result,
            'env': env if env else self.getEnv(),
        }
        document = Database.SimulationResult(documentDict, backend=self.backend)
        document.save()
        self.backend.commit()

    def filter(self, *args, **kwargs):
        return self.backend.filter(Database.SimulationResult, *args, **kwargs)

    def alreadySimulated(self, parameters):
        return len(self.filter({'params': parameters})) > 0

    def toPandas(self, query):
        queryResult = self.backend.filter(self.SimulationResult, query)
        records = []
        index = set()
        for e in queryResult:
            record = OrderedDict(e.params.items())
            record.update(e.result)
            records.append(record)
            index.update(e.params.keys())

        df = pd.DataFrame.from_records(records)

        return df


