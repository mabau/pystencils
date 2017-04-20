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


class Database:
    class SimulationResult(blitzdb.Document):
        pass

    def __init__(self, file):
        self.backend = blitzdb.FileBackend(file)
        self.backend.autocommit = True

    def save(self, params, result, env={}):
        env = env.copy()
        env['timestamp'] = time.mktime(time.gmtime())
        env['hostname'] = socket.gethostname()
        env['cpuCompilerConfig'] = getCompilerConfig()
        documentDict = {
            'params': params,
            'result': result,
            'env': env,
        }
        document = Database.SimulationResult(documentDict, backend=self.backend)
        document.save()
        self.backend.commit()

    def filter(self, **kwargs):
        self.backend.filter(Database.SimulationResult, **kwargs)

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

        #df.set_index([df[a] for a in list(index)], drop=True, inplace=True)
        #for ind in index:
        #    del df[ind]

        #df.set_index([getattr(df, n) for n in index], inplace=True)
        #df.set_index(tuple(index), inplace=True)
        #df.set_index([df[col] for col in index], inplace=True)
        return df

