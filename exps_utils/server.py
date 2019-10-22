import json
import argparse
try:
    import SocketServer as socketserver
except ImportError:
    import socketserver

import shlex
import filelock
from pathlib import Path

# https://docs.python.org/3/library/socketserver.html#socketserver-tcpserver-example
# https://stackoverflow.com/questions/10810249/python-socket-multiple-clients

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='')
    parser.add_argument('--port', type=int, default=9999)
    return parser.parse_args(args)

def read_args_server(args_path, timeout=30):
    args_path = Path(args_path)
    lock_dir = Path(args_path.parent, '.lock')
    if not lock_dir.exists():
        lock_dir.mkdir(parents=True)
    lock_fn = Path(lock_dir, args_path.stem)
    lock_fn.touch(exist_ok=True)
    with filelock.FileLock(str(lock_fn)).acquire(timeout=timeout):
        with open(str(args_path)) as f:
            jobs = f.read().splitlines(True) # same as convert_arg_line_to_args
        while jobs:
            job = jobs[0].strip()
            if not job or job.startswith('#'):
                jobs = jobs[1:]
            else:
                break
        if jobs:
            # skip empty line and comments
            args = shlex.split(jobs[0])
            with open(str(args_path), 'w') as f:
                f.writelines(jobs[1:])
        else:
            args = None
    return args

def push_args_server(args_str, args_path, timeout=30):
    args_path = Path(args_path)
    lock_dir = Path(args_path.parent, '.lock')
    if not lock_dir.exists():
        lock_dir.mkdir(parents=True)
    lock_fn = Path(lock_dir, args_path.stem)
    lock_fn.touch(exist_ok=True) # disadvantages: this will not be cleaned up
    with filelock.FileLock(str(lock_fn)).acquire(timeout=timeout):
        with open(str(args_path)) as f:
            jobs = f.read().splitlines(True)
        jobs.insert(0, args_str + '\n')
        with open(str(args_path), 'w') as f:
            f.writelines(jobs)
    return True # success operation

class TCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        data = self.request.recv(1024).strip()
        data = json.loads(data)
        print('request from {}'.format(self.client_address[0]))
        if data['request'] == 'read_args':
            reply = {
                'args': read_args_server(data['args_path']),
            }
        elif data['request'] == 'push_args':
            reply = {
                'success': push_args_server(
                    data['args_str'],
                    data['args_path'], 
                ),
            }
        else:
            reply = 'ignore unknown request: {}'.format(data)
            print(reply)
        self.request.sendall(bytes(json.dumps(reply), 'utf-8'))

def run_server(args=None):
    args = parse_args(args)
    print('start exp server')
    with socketserver.TCPServer((args.host, args.port), TCPHandler) as server:
        server.serve_forever()

if __name__ == '__main__':
    run_server()
