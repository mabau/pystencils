import json
import datetime
import os
import socket
import itertools
from copy import deepcopy
from collections import namedtuple
from time import sleep
from pystencils.runhelper import Database
from pystencils.utils import DotDict


class ParameterStudy(object):
    Run = namedtuple("Run", ['parameterDict', 'weight'])

    def __init__(self, runFunction, listOfRuns=[], databaseFile='./db'):
        self.listOfRuns = listOfRuns
        self.runFunction = runFunction
        self.db = Database(databaseFile)

    def addRun(self, parameterDict, weight=1):
        self.listOfRuns.append(self.Run(parameterDict, weight))

    def addCombinations(self, degreesOfFreedom, constantParameters=None, filterFunction=None, weightFunction=None):
        parameterNames = [e[0] for e in degreesOfFreedom]
        parameterValues = [e[1] for e in degreesOfFreedom]

        defaultParamsDict = {} if constantParameters is None else constantParameters
        for valueTuple in itertools.product(*parameterValues):
            paramsDict = deepcopy(defaultParamsDict)
            paramsDict.update({name: value for name, value in zip(parameterNames, valueTuple)})
            params = DotDict(paramsDict)
            if filterFunction:
                params = filterFunction(params)
                if params is None:
                    continue
            weight = 1 if not weightFunction else weightFunction(params)
            self.addRun(params, weight)

    def filterAlreadySimulated(self, allRuns):
        return [r for r in allRuns if not self.db.alreadySimulated(r.parameterDict)]

    @staticmethod
    def distributeRuns(allRuns, process, numProcesses):
        sortedRuns = sorted(allRuns, key=lambda e: e.weight, reverse=True)
        result = sortedRuns[process::numProcesses]
        result.reverse()  # start with faster scenarios
        return result

    def runServer(self, ip="0.0.0.0", port=8342):
        from http.server import BaseHTTPRequestHandler, HTTPServer
        filteredRuns = self.filterAlreadySimulated(self.listOfRuns)

        if not filteredRuns:
            print("No Scenarios to simulate")
            return

        class ParameterStudyServer(BaseHTTPRequestHandler):
            parameterStudy = self
            allRuns = filteredRuns
            runs = filteredRuns.copy()
            currentlyRunning = {}
            finishedRuns = []

            def nextScenario(self, receivedJsonData):
                clientName = receivedJsonData['clientName']
                if len(self.runs) > 0:
                    runStatus = "%d/%d" % (len(self.finishedRuns), len(self.allRuns))
                    workStatus = "%d/%d" % (sum(r.weight for r in self.finishedRuns),
                                            sum(r.weight for r in self.allRuns))
                    formatArgs = {
                        'remaining': len(self.runs),
                        'time': datetime.datetime.now().strftime("%H:%M:%S"),
                        'clientName': clientName,
                        'runStatus': runStatus,
                        'workStatus': workStatus,
                    }

                    scenario = self.runs.pop(0)
                    print(" {time} {clientName} fetched scenario. Scenarios: {runStatus}, Work: {workStatus}"
                          .format(**formatArgs))
                    self.currentlyRunning[clientName] = scenario
                    return {'status': 'ok', 'params': scenario.parameterDict}
                else:
                    return {'status': 'finished'}

            def result(self, receivedJsonData):
                clientName = receivedJsonData['clientName']
                run = self.currentlyRunning[clientName]
                self.finishedRuns.append(run)
                del self.currentlyRunning[clientName]
                d = receivedJsonData

                def hash_dict(d):
                    import hashlib
                    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()

                assert hash_dict(d['params']) == hash_dict(run.parameterDict)
                self.parameterStudy.db.save(run.parameterDict, d['result'], d['env'], changedParams=d['changedParams'])
                return {}

            def do_POST(self):
                mapping = {'/nextScenario': self.nextScenario,
                           '/result': self.result}
                if self.path in mapping.keys():
                    data = self.rfile.read(int(self.headers['Content-Length']))
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    jsonData = json.loads(data.decode())
                    response = mapping[self.path](jsonData)
                    self.wfile.write(json.dumps(response).encode())
                else:
                    self.send_response(400)

            def do_GET(self):
                return self.do_POST()

            def log_message(self, format, *args):
                return

        print("Listening to connections on {}:{}. Scenarios to simulate: {}".format(ip, port, len(filteredRuns)))
        server = HTTPServer((ip, port), ParameterStudyServer)
        while len(ParameterStudyServer.currentlyRunning) > 0 or len(ParameterStudyServer.runs) > 0:
            server.handle_request()
        server.handle_request()

    def runClient(self, clientName="{hostname}_{pid}", server='localhost', port=8342, parameterUpdate={}):
        from urllib.request import urlopen, URLError
        url = "http://{}:{}".format(server, port)
        clientName = clientName.format(hostname=socket.gethostname(), pid=os.getpid())
        while True:
            try:
                httpResponse = urlopen(url + "/nextScenario",
                                       data=json.dumps({'clientName': clientName}).encode())
                scenario = json.loads(httpResponse.read().decode())
                if scenario['status'] != 'ok':
                    break
                originalParams = scenario['params'].copy()
                scenario['params'].update(parameterUpdate)
                result = self.runFunction(**scenario['params'])

                answer = {'params': originalParams,
                          'changedParams': parameterUpdate,
                          'result': result,
                          'env': Database.getEnv(),
                          'clientName': clientName}
                urlopen(url + '/result', data=json.dumps(answer).encode())
            except URLError:
                print("Cannot connect to server {}  retrying in 5 seconds...".format(url))
                sleep(5)

    def run(self, process, numProcesses, parameterUpdate={}):
        ownRuns = self.distributeRuns(self.listOfRuns, process, numProcesses)
        for run in ownRuns:
            parameterDict = run.parameterDict.copy()
            parameterDict.update(parameterUpdate)
            result = self.runFunction(**parameterDict)

            self.db.save(run.parameterDict, result, None, changedParams=parameterUpdate)

    def runScenariosNotInDatabase(self, parameterUpdate={}):
        filteredRuns = self.filterAlreadySimulated(self.listOfRuns)
        for run in filteredRuns:
            parameterDict = run.parameterDict.copy()
            parameterDict.update(parameterUpdate)
            result = self.runFunction(**parameterDict)

            self.db.save(run.parameterDict, result, None, changedParams=parameterUpdate)

    def runFromCommandLine(self, argv=None):
        from argparse import ArgumentParser

        def server(a):
            if a.database:
                self.db = Database(a.database)
            self.runServer(a.host, a.port)

        def client(a):
            self.runClient(a.clientName, a.host, a.port, json.loads(a.parameterOverride))

        def local(a):
            if a.database:
                self.db = Database(a.database)
            self.runScenariosNotInDatabase(json.loads(a.parameterOverride))

        parser = ArgumentParser()
        subparsers = parser.add_subparsers()

        localParser = subparsers.add_parser('local', aliases=['l'],
                                            help="Run scenarios locally which are not yet in database",)
        localParser.add_argument("-d", "--database", type=str, default="")
        localParser.add_argument("-P", "--parameterOverride", type=str, default="{}",
                                 help="JSON: the parameter dictionary is updated with these parameters. Use this to "
                                      "set host specific options like GPU call parameters. Enclose in \" ")
        localParser.set_defaults(func=local)

        serverParser = subparsers.add_parser('server', aliases=['serv', 's'],
                                             help="Runs server to distribute different scenarios to workers",)
        serverParser.add_argument("-p", "--port", type=int, default=8342, help="Port to listen on")
        serverParser.add_argument("-H", "--host", type=str, default="0.0.0.0", help="IP/Hostname to listen on")
        serverParser.add_argument("-d", "--database", type=str, default="")
        serverParser.set_defaults(func=server)

        clientParser = subparsers.add_parser('client', aliases=['c'],
                                             help="Runs a worker client connection to scenario distribution server")
        clientParser.add_argument("-p", "--port", type=int, default=8342, help="Port to connect to")
        clientParser.add_argument("-H", "--host", type=str, default="localhost", help="Host or IP to connect to")
        clientParser.add_argument("-n", "--clientName", type=str, default="{hostname}_{pid}",
                                  help="Unique client name, you can use {hostname} and {pid} as placeholder")
        clientParser.add_argument("-P", "--parameterOverride", type=str, default="{}",
                                  help="JSON: the parameter dictionary is updated with these parameters. Use this to "
                                       "set host specific options like GPU call parameters. Enclose in \" ")
        clientParser.set_defaults(func=client)

        args = parser.parse_args(argv)
        if not len(vars(args)):
            parser.print_help()
        else:
            args.func(args)

