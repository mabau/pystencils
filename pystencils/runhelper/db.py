import socket
import time
from typing import Dict, Iterator, Sequence

import blitzdb

from pystencils.cpu.cpujit import get_compiler_config


class Database:
    """NoSQL database for storing simulation results.

    Two backends are supported:
        * `blitzdb`: simple file-based solution similar to sqlite for SQL databases, stores json files
                     no server setup required, but slow for larger collections
        * `mongodb`: mongodb backend via `pymongo`

    A simulation result is stored as an object consisting of
        * parameters: dict with simulation parameters
        * results: dict with results
        * environment: information about the machine, compiler configuration and time

    Args:
        file: database identifier, for blitzdb pass a directory name here. Database folder is created if it doesn't
              exist yet. For larger collections use mongodb. In this case pass a pymongo connection string
              e.g. "mongo://server:9131"

    Example:
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as tmp_dir:
        ...     db = Database(tmp_dir)  # create database in temporary folder
        ...     params = {'method': 'finite_diff', 'dx': 1.5}  # some hypothetical simulation parameters
        ...     db.save(params, result={'error': 1e-6})  # save simulation parameters together with hypothetical results
        ...     assert db.was_already_simulated(params)  # search for parameters in database
        ...     assert next(db.filter_params(params))['params'] == params # get data set, keys are 'params', 'results'
        ...                                                               # and 'env'
        ...     # get a pandas object with all results matching a query
        ...     df = db.to_pandas({'dx': 1.5}, remove_prefix=True)
        ...     # order columns alphabetically (just for doctest output)
        ...     df.reindex(sorted(df.columns), axis=1)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
                                 dx     error       method
                pk
                ...             1.5  0.000001  finite_diff
    """

    class SimulationResult(blitzdb.Document):
        pass

    def __init__(self, file: str) -> None:
        if file.startswith("mongo://"):
            from pymongo import MongoClient
            db_name = file[len("mongo://"):]
            c = MongoClient()
            self.backend = blitzdb.MongoBackend(c[db_name])
        else:
            self.backend = blitzdb.FileBackend(file)

        self.backend.autocommit = True

    def save(self, params: Dict, result: Dict, env: Dict = None, **kwargs) -> None:
        """Stores a simulation result in the database.

        Args:
            params: dict of simulation parameters
            result: dict of simulation results
            env: optional environment - if None a default environment with compiler configuration, machine info and time
                 is used
            **kwargs: the final object is updated with the keyword arguments

        """
        document_dict = {
            'params': params,
            'result': result,
            'env': env if env else self.get_environment(),
        }
        document_dict.update(kwargs)
        document = Database.SimulationResult(document_dict, backend=self.backend)
        document.save()
        self.backend.commit()

    def filter_params(self, parameter_query: Dict, *args, **kwargs) -> Iterator['SimulationResult']:
        """Query using simulation parameters.

        See blitzdb documentation for filter

        Args:
            parameter_query: blitzdb filter dict using only simulation parameters
            *args: arguments passed to blitzdb filter
            **kwargs: arguments passed to blitzdb filter

        Returns:
            generator of SimulationResult, which is a dict-like object with keys 'params', 'result' and 'env'
        """
        query = {'params.' + k: v for k, v in parameter_query.items()}
        return self.filter(query, *args, **kwargs)

    def filter(self, *args, **kwargs):
        """blitzdb filter on SimulationResult, not only simulation parameters.

        Can be used to filter for results or environment options.
        The filter dictionary has to have prefixes "params." , "env." or "result."
        """
        return self.backend.filter(Database.SimulationResult, *args, **kwargs)

    def was_already_simulated(self, parameters):
        """Checks if there is at least one simulation result matching the passed parameters."""
        return len(self.filter({'params': parameters})) > 0

    # Columns with these prefixes are not included in pandas result
    pandas_columns_to_ignore = ['changedParams.', 'env.']

    def to_pandas(self, parameter_query, remove_prefix=True, drop_constant_columns=False):
        """Queries for simulations with given parameters and returns them in a pandas data frame.

        Args:
            parameter_query: see filter method
            remove_prefix: if True the name of the pandas columns are not prefixed with "params." or "results."
            drop_constant_columns: if True, all columns are dropped that have the same value is all rows

        Returns:
            pandas data frame
        """
        from pandas import json_normalize

        query_result = self.filter_params(parameter_query)
        attributes = [e.attributes for e in query_result]
        if not attributes:
            return
        df = json_normalize(attributes)
        df.set_index('pk', inplace=True)

        if self.pandas_columns_to_ignore:
            remove_columns_by_prefix(df, self.pandas_columns_to_ignore, inplace=True)
        if remove_prefix:
            remove_prefix_in_column_name(df, inplace=True)
        if drop_constant_columns:
            df, _ = remove_constant_columns(df)

        return df

    @staticmethod
    def get_environment():
        result = {
            'timestamp': time.mktime(time.gmtime()),
            'hostname': socket.gethostname(),
            'cpuCompilerConfig': get_compiler_config(),
        }
        try:
            from git import Repo, InvalidGitRepositoryError
            repo = Repo(search_parent_directories=True)
            result['git_hash'] = str(repo.head.commit)
        except (ImportError, InvalidGitRepositoryError):
            pass

        return result

# ----------------------------------------- Helper Functions -----------------------------------------------------------


def remove_constant_columns(df):
    """Removes all columns of a pandas data frame that have the same value in all rows."""
    import pandas as pd
    remaining_df = df.loc[:, df.apply(pd.Series.nunique) > 1]
    constants = df.loc[:, df.apply(pd.Series.nunique) <= 1].iloc[0]
    return remaining_df, constants


def remove_columns_by_prefix(df, prefixes: Sequence[str], inplace: bool = False):
    """Remove all columns from a pandas data frame whose name starts with one of the given prefixes."""
    if not inplace:
        df = df.copy()

    for column_name in df.columns:
        for prefix in prefixes:
            if column_name.startswith(prefix):
                del df[column_name]
    return df


def remove_prefix_in_column_name(df, inplace: bool = False):
    """Removes dotted prefixes from pandas column names.

    A column named 'result.finite_diff.dx' is renamed to 'finite_diff.dx', everything before the first dot is removed.
    If the column name does not contain a dot, the column name is not changed.
    """
    if not inplace:
        df = df.copy()

    new_column_names = []
    for column_name in df.columns:
        if '.' in column_name:
            new_column_names.append(column_name[column_name.index('.') + 1:])
        else:
            new_column_names.append(column_name)
    df.columns = new_column_names
    return df
