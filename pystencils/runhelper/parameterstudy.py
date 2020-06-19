import datetime
import itertools
import json
import os
import socket
from collections import namedtuple
from copy import deepcopy
from time import sleep
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from pystencils.runhelper import Database
from pystencils.utils import DotDict

ParameterDict = Dict[str, Any]
WeightFunction = Callable[[Dict], int]
FilterFunction = Callable[[ParameterDict], Optional[ParameterDict]]


class ParameterStudy:
    """Manages and runs multiple configurations locally or distributed and stores results in NoSQL database.

    To run a parameter study, define a run function that takes all parameters as keyword arguments and returns the
    results as a (possibly nested) dictionary. Then, define the parameter sets that this function should be run with.

    Examples:
          >>> import tempfile
          >>>
          >>> def dummy_run_function(p1, p2, p3, p4):
          ...     print("Run called with", p1, p2, p3, p4)
          ...     return { 'result1': p1 * p2, 'result2': p3 + p4 }
          >>>
          >>> with tempfile.TemporaryDirectory() as tmp_dir:
          ...     ps = ParameterStudy(dummy_run_function, database_connector=tmp_dir)
          ...     ps.add_run({'p1': 5, 'p2': 42, 'p3': 'abc', 'p4': 'def'})
          ...     ps.add_combinations( [('p1', [1, 2]),
          ...                           ('p3', ['x', 'y'])], constant_parameters={'p2': 5, 'p4': 'z' })
          ...     ps.run()
          ...     ps.run_scenarios_not_in_database()
          ...     ps.run_from_command_line(argv=['local'])  # alternative to run - exposes a command line interface if
          ...                                               # no argv is passed. Does not run anything here, because
          ...                                               # configuration already in database are skipped
          Run called with 2 5 y z
          Run called with 2 5 x z
          Run called with 1 5 y z
          Run called with 1 5 x z
          Run called with 5 42 abc def

    Above example runs all parameter combinations locally and stores the returned result in the NoSQL database.
    It is also possible to distribute the runs to multiple processes, by starting a server on one machine and multiple
    executing runners on other machines. The server distributes configurations to the runners, collects their results
    to stores the results in the database.
    """

    Run = namedtuple("Run", ['parameter_dict', 'weight'])

    def __init__(self, run_function: Callable[..., Dict], runs: Sequence = (),
                 database_connector: str = './db') -> None:
        self.runs = list(runs)
        self.run_function = run_function
        self.db = Database(database_connector)

    def add_run(self, parameter_dict: ParameterDict, weight: int = 1) -> None:
        """Schedule a dictionary of parameters to run in this parameter study.

        Args:
            parameter_dict: used as keyword arguments to the run function.
            weight: weight of the run configuration which should be  proportional to runtime of this case,
                    used for progress display and distribution to processes.
        """
        self.runs.append(self.Run(parameter_dict, weight))

    def add_combinations(self, degrees_of_freedom: Sequence[Tuple[str, Sequence[Any]]],
                         constant_parameters: Optional[ParameterDict] = None,
                         filter_function: Optional[FilterFunction] = None,
                         runtime_weight_function: Optional[WeightFunction] = None) -> None:
        """Add all possible combinations of given parameters as runs.

        This is a convenience function to simulate all possible parameter combinations of a scenario.
        Configurations can be filtered and weighted by passing filter- and weighting functions.

        Args:
            degrees_of_freedom: defines for each parameter the possible values it can take on
            constant_parameters: parameter dict, for parameters that should not be changed
            filter_function: optional function that receives a parameter dict and returns the potentially modified dict
                             or None if this combination should not be added.
            runtime_weight_function: function mapping a parameter dict to the runtime weight (see weight at add_runs)

        Examples:
             degrees_of_freedom = [('p1', [1,2]),
                                   ('p2', ['a', 'b'])]
             is equivalent to calling add_run four times, with all possible parameter combinations.
        """
        parameter_names = [e[0] for e in degrees_of_freedom]
        parameter_values = [e[1] for e in degrees_of_freedom]

        default_params_dict = {} if constant_parameters is None else constant_parameters
        for value_tuple in itertools.product(*parameter_values):
            params_dict = deepcopy(default_params_dict)
            params_dict.update({name: value for name, value in zip(parameter_names, value_tuple)})
            params = DotDict(params_dict)
            if filter_function:
                params = filter_function(params)
                if params is None:
                    continue
            weight = 1 if not runtime_weight_function else runtime_weight_function(params)
            self.add_run(params, weight)

    def run(self, process: int = 0, num_processes: int = 1, parameter_update: Optional[ParameterDict] = None) -> None:
        """Runs all added configurations.

        Args:
            process: configurations are split into num_processes chunks according to weights and only the
                     process'th chunk is run. To run all, use process=0 and num_processes=1
            num_processes: see above
            parameter_update: Extend/override all configurations with this dictionary.
        """
        parameter_update = {} if parameter_update is None else parameter_update
        own_runs = self._distribute_runs(self.runs, process, num_processes)
        for run in own_runs:
            parameter_dict = run.parameter_dict.copy()
            parameter_dict.update(parameter_update)
            result = self.run_function(**parameter_dict)

            self.db.save(run.parameter_dict, result, None, changed_params=parameter_update)

    def run_scenarios_not_in_database(self, parameter_update: Optional[ParameterDict] = None) -> None:
        """Same as run method, but runs only configuration for which no result is in the database yet."""
        parameter_update = {} if parameter_update is None else parameter_update
        filtered_runs = self._filter_already_simulated(self.runs)
        for run in filtered_runs:
            parameter_dict = run.parameter_dict.copy()
            parameter_dict.update(parameter_update)
            result = self.run_function(**parameter_dict)

            self.db.save(run.parameter_dict, result, changed_params=parameter_update)

    def run_server(self, ip: str = "0.0.0.0", port: int = 8342):
        """Runs server to supply runner clients with scenarios to simulate and collect results from them.
        Skips scenarios that are already in the database."""
        from http.server import BaseHTTPRequestHandler, HTTPServer
        filtered_runs = self._filter_already_simulated(self.runs)

        if not filtered_runs:
            print("No Scenarios to simulate")
            return

        class ParameterStudyServer(BaseHTTPRequestHandler):
            parameterStudy = self
            all_runs = filtered_runs
            runs = filtered_runs.copy()
            currently_running = {}
            finished_runs = []

            def next_scenario(self, received_json_data):
                client_name = received_json_data['client_name']
                if len(self.runs) > 0:
                    run_status = "%d/%d" % (len(self.finished_runs), len(self.all_runs))
                    work_status = "%d/%d" % (sum(r.weight for r in self.finished_runs),
                                             sum(r.weight for r in self.all_runs))
                    format_args = {
                        'remaining': len(self.runs),
                        'time': datetime.datetime.now().strftime("%H:%M:%S"),
                        'client_name': client_name,
                        'run_status': run_status,
                        'work_status': work_status,
                    }

                    scenario = self.runs.pop(0)
                    print(" {time} {client_name} fetched scenario. Scenarios: {run_status}, Work: {work_status}"
                          .format(**format_args))
                    self.currently_running[client_name] = scenario
                    return {'status': 'ok', 'params': scenario.parameter_dict}
                else:
                    return {'status': 'finished'}

            def result(self, received_json_data):
                client_name = received_json_data['client_name']
                run = self.currently_running[client_name]
                self.finished_runs.append(run)
                del self.currently_running[client_name]
                d = received_json_data

                def hash_dict(dictionary):
                    import hashlib
                    return hashlib.sha1(json.dumps(dictionary, sort_keys=True).encode()).hexdigest()

                assert hash_dict(d['params']) == hash_dict(run.parameter_dict), \
                    str(d['params']) + "is not equal to " + str(run.parameter_dict)
                self.parameterStudy.db.save(run.parameter_dict,
                                            result=d['result'], env=d['env'], changed_params=d['changed_params'])
                return {}

            # noinspection PyPep8Naming
            def do_POST(self) -> None:
                mapping = {'/next_scenario': self.next_scenario,
                           '/result': self.result}
                if self.path in mapping.keys():
                    data = self._read_contents()
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    json_data = json.loads(data)
                    response = mapping[self.path](json_data)
                    self.wfile.write(json.dumps(response).encode())
                else:
                    self.send_response(400)

            # noinspection PyPep8Naming
            def do_GET(self):
                return self.do_POST()

            def _read_contents(self):
                return self.rfile.read(int(self.headers['Content-Length'])).decode()

            def log_message(self, fmt, *args):
                return

        print(f"Listening to connections on {ip}:{port}. Scenarios to simulate: {len(filtered_runs)}")
        server = HTTPServer((ip, port), ParameterStudyServer)
        while len(ParameterStudyServer.currently_running) > 0 or len(ParameterStudyServer.runs) > 0:
            server.handle_request()
        server.handle_request()

    def run_client(self, client_name: str = "{hostname}_{pid}", server: str = 'localhost', port: int = 8342,
                   parameter_update: Optional[ParameterDict] = None, max_time=None) -> None:
        """Start runner client that retrieves configuration from server, runs it and reports results back to server.

        Args:
            client_name: name of the client. Has to be unique for each client.
                         Placeholders {hostname} and {pid} can be used to generate unique name.
            server: url to server
            port: port as specified in run_server
            parameter_update: Used to override/extend parameters received from the server.
                              Typical use cases is to set optimization or GPU parameters for some clients to make
                              some clients simulate on CPU, others on GPU
            max_time: maximum runtime in seconds: the client runs scenario after scenario, but starts only a new
                      scenario if not more than max_time seconds have passed since this function was called.
                      So the time given here should be the total maximum runtime minus a typical runtime for one setup
        """
        from urllib.request import urlopen
        from urllib.error import URLError
        import time
        parameter_update = {} if parameter_update is None else parameter_update
        url = f"http://{server}:{port}"
        client_name = client_name.format(hostname=socket.gethostname(), pid=os.getpid())
        start_time = time.time()
        while True:
            try:
                if max_time is not None and (time.time() - start_time) > max_time:
                    print("Stopping client - maximum time reached")
                    break
                http_response = urlopen(url + "/next_scenario",
                                        data=json.dumps({'client_name': client_name}).encode())
                scenario = json.loads(http_response.read().decode())
                if scenario['status'] != 'ok':
                    break
                original_params = scenario['params'].copy()
                scenario['params'].update(parameter_update)
                result = self.run_function(**scenario['params'])

                answer = {'params': original_params,
                          'changed_params': parameter_update,
                          'result': result,
                          'env': Database.get_environment(),
                          'client_name': client_name}
                urlopen(url + '/result', data=json.dumps(answer).encode())
            except URLError:
                print(f"Cannot connect to server {url}  retrying in 5 seconds...")
                sleep(5)

    def run_from_command_line(self, argv: Optional[Sequence[str]] = None) -> None:
        """Exposes interface to command line with possibility to run directly or distributed via server/client."""
        from argparse import ArgumentParser

        def server(a):
            if a.database:
                self.db = Database(a.database)
            self.run_server(a.host, a.port)

        def client(a):
            self.run_client(a.client_name, a.host, a.port, json.loads(a.parameter_override), a.max_time)

        def local(a):
            if a.database:
                self.db = Database(a.database)
            self.run_scenarios_not_in_database(json.loads(a.parameter_override))

        parser = ArgumentParser()
        subparsers = parser.add_subparsers()

        local_parser = subparsers.add_parser('local', aliases=['l'],
                                             help="Run scenarios locally which are not yet in database", )
        local_parser.add_argument("-d", "--database", type=str, default="")
        local_parser.add_argument("-P", "--parameter_override", type=str, default="{}",
                                  help="JSON: the parameter dictionary is updated with these parameters. Use this to "
                                       "set host specific options like GPU call parameters. Enclose in \" ")
        local_parser.set_defaults(func=local)

        server_parser = subparsers.add_parser('server', aliases=['s'],
                                              help="Runs server to distribute different scenarios to workers", )
        server_parser.add_argument("-p", "--port", type=int, default=8342, help="Port to listen on")
        server_parser.add_argument("-H", "--host", type=str, default="0.0.0.0", help="IP/Hostname to listen on")
        server_parser.add_argument("-d", "--database", type=str, default="")
        server_parser.set_defaults(func=server)

        client_parser = subparsers.add_parser('client', aliases=['c'],
                                              help="Runs a worker client connection to scenario distribution server")
        client_parser.add_argument("-p", "--port", type=int, default=8342, help="Port to connect to")
        client_parser.add_argument("-H", "--host", type=str, default="localhost", help="Host or IP to connect to")
        client_parser.add_argument("-n", "--client_name", type=str, default="{hostname}_{pid}",
                                   help="Unique client name, you can use {hostname} and {pid} as placeholder")
        client_parser.add_argument("-P", "--parameter_override", type=str, default="{}",
                                   help="JSON: the parameter dictionary is updated with these parameters. Use this to "
                                        "set host specific options like GPU call parameters. Enclose in \" ")
        client_parser.add_argument("-t", "--max_time", type=int, default=None,
                                   help="If more than this time in seconds has passed, "
                                        "the client stops running scenarios.")
        client_parser.set_defaults(func=client)

        args = parser.parse_args(argv)
        if not len(vars(args)):
            parser.print_help()
        else:
            args.func(args)

    def _filter_already_simulated(self, all_runs):
        """Removes all runs from the given list, that are already in the database"""
        already_simulated = {json.dumps(e.params) for e in self.db.filter({})}
        return [r for r in all_runs if json.dumps(r.parameter_dict) not in already_simulated]

    @staticmethod
    def _distribute_runs(all_runs, process, num_processes):
        """Partitions runs by their weights into num_processes chunks and returns the process's chunk."""
        sorted_runs = sorted(all_runs, key=lambda e: e.weight, reverse=True)
        result = sorted_runs[process::num_processes]
        result.reverse()  # start with faster scenarios
        return result
