import time
import socket
import blitzdb
from pystencils.cpu.cpujit import getCompilerConfig


def removeConstantColumns(df):
    import pandas as pd
    remainingDf = df.loc[:, df.apply(pd.Series.nunique) > 1]
    constants = df.loc[:, df.apply(pd.Series.nunique) <= 1].iloc[0]
    return remainingDf, constants


def removeColumnsByPrefix(df, prefixes, inplace=False):
    if not inplace:
        df = df.copy()

    for columnName in df.columns:
        for prefix in prefixes:
            if columnName.startswith(prefix):
                del df[columnName]
    return df


def removePrefixInColumnName(df, inplace=False):
    if not inplace:
        df = df.copy()

    newColumnNames = []
    for columnName in df.columns:
        if '.' in columnName:
            newColumnNames.append(columnName[columnName.index('.') + 1:])
        else:
            newColumnNames.append(columnName)
    df.columns = newColumnNames
    return df


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

    def save(self, params, result, env=None, **kwargs):
        documentDict = {
            'params': params,
            'result': result,
            'env': env if env else self.getEnv(),
        }
        documentDict.update(kwargs)
        document = Database.SimulationResult(documentDict, backend=self.backend)
        document.save()
        self.backend.commit()

    def filter(self, *args, **kwargs):
        return self.backend.filter(Database.SimulationResult, *args, **kwargs)

    def filterParams(self, query, *args, **kwargs):
        query = {'params.' + k: v for k, v in query.items()}
        return self.filter(query, *args, **kwargs)

    def alreadySimulated(self, parameters):
        return len(self.filter({'params': parameters})) > 0

    # Columns with these prefixes are not included in pandas result
    pandasColumnsToIgnore = ['changedParams.', 'env.']

    def toPandas(self, parameterQuery, removePrefix=True, dropConstantColumns=False):
        import pandas as pd

        queryResult = self.filterParams(parameterQuery)
        if len(queryResult) == 0:
            return

        df = pd.io.json.json_normalize([e.attributes for e in queryResult])
        df.set_index('pk', inplace=True)

        if self.pandasColumnsToIgnore:
            removeColumnsByPrefix(df, self.pandasColumnsToIgnore, inplace=True)
        if removePrefix:
            removePrefixInColumnName(df, inplace=True)
        if dropConstantColumns:
            df, _ = removeConstantColumns(df)

        return df
