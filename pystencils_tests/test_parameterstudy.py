import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from tempfile import TemporaryDirectory

from pystencils.runhelper import ParameterStudy


def test_http_server(monkeypatch):

    result_list = []

    def handle_request_mock(server):
        handler = server.RequestHandlerClass()

        def get(url, data):
            handler.wfile = io.BytesIO()
            handler.path = url
            handler._read_contents = lambda *args, **kwargs: json.dumps(data)
            handler.do_GET()
            handler.wfile.seek(0)
            return json.loads(handler.wfile.read().decode())

        while True:
            result = get('/next_scenario', {'client_name': 'test'})
            if result['status'] == 'finished':
                break
            else:
                assert result['status'] == 'ok'
                result_list.append(result)

                p = result['params']
                get("/result", {'params': p,
                                'changed_params': {},
                                'result': {'result': p['p1'] + p['p2']},
                                'env': {},
                                'client_name': 'test'})

    monkeypatch.setattr(HTTPServer, 'handle_request', handle_request_mock)
    monkeypatch.setattr(BaseHTTPRequestHandler, '__init__', lambda self: None)
    monkeypatch.setattr(BaseHTTPRequestHandler, 'send_response', lambda *args, **kwargs: None)
    monkeypatch.setattr(BaseHTTPRequestHandler, 'send_header', lambda *args, **kwargs: None)
    monkeypatch.setattr(BaseHTTPRequestHandler, 'end_headers', lambda *args, **kwargs: None)

    with TemporaryDirectory() as tmp_dir:
        ps = ParameterStudy(lambda p1, p2: p1 + p2, database_connector=tmp_dir)
        ps.add_combinations([('p1', [1, 2])], constant_parameters={'p2': 3})
        ps.run_server()
        assert len(result_list) == 2


def test_http_client(monkeypatch):
    import urllib.request

    call_count = 0

    def simulation_dummy(p1, p2):
        nonlocal call_count
        call_count += 1

    answers = [{"status": 'ok', "params": {'p1': 1, 'p2': 2}}, {},
               {'status': 'finished'}, ]
    next_answer = 0

    def urlopen_mock(_, data):
        nonlocal next_answer
        data = data.decode()
        assert 'client_name' in data
        result = io.BytesIO(json.dumps(answers[next_answer]).encode())
        next_answer += 1
        return result

    monkeypatch.setattr(urllib.request, 'urlopen', urlopen_mock)

    with TemporaryDirectory() as tmp_dir:
        ps = ParameterStudy(simulation_dummy, database_connector=tmp_dir)
        ps.run_client('some_name')

    assert call_count == 1
