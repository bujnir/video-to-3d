"""Tiny job server that runs on the RunPod GPU pod.

Endpoints (all need header X-Token matching env JOB_TOKEN):
  PUT  /f/<path>   upload a file to /workspace/<path>
  GET  /f/<path>   download /workspace/<path>
  POST /run        body = bash script; runs in background, output to /workspace/job.log
  GET  /status     {"running": bool, "exit": code or null}
  GET  /log?n=80   last n lines of the job log
"""
import http.server, json, os, subprocess, threading, urllib.parse

ROOT = '/workspace'
TOKEN = os.environ.get('JOB_TOKEN', '')
state = {'proc': None}


class H(http.server.BaseHTTPRequestHandler):
    def _ok(self):
        if not TOKEN or self.headers.get('X-Token') != TOKEN:
            self.send_response(403); self.end_headers(); return False
        return True

    def _send(self, code, body, ctype='application/json'):
        if isinstance(body, (dict, list)):
            body = json.dumps(body)
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _path(self):
        rel = urllib.parse.unquote(urllib.parse.urlparse(self.path).path[3:])
        p = os.path.realpath(os.path.join(ROOT, rel))
        return p if p.startswith(ROOT + '/') else None

    def do_PUT(self):
        if not self._ok(): return
        p = self._path()
        if not p: return self._send(400, {'error': 'bad path'})
        os.makedirs(os.path.dirname(p), exist_ok=True)
        n = int(self.headers.get('Content-Length', 0))
        with open(p, 'wb') as f:
            while n > 0:
                chunk = self.rfile.read(min(n, 1 << 20)); f.write(chunk); n -= len(chunk)
        self._send(200, {'saved': p, 'bytes': os.path.getsize(p)})

    def do_GET(self):
        if self.path == '/health':
            return self._send(200, {'ok': True})
        if not self._ok(): return
        u = urllib.parse.urlparse(self.path)
        if u.path.startswith('/f/'):
            p = self._path()
            if not p or not os.path.isfile(p): return self._send(404, {'error': 'not found'})
            with open(p, 'rb') as f: data = f.read()
            return self._send(200, data, 'application/octet-stream')
        if u.path == '/status':
            pr = state['proc']
            return self._send(200, {'running': bool(pr and pr.poll() is None),
                                    'exit': None if not pr else pr.poll()})
        if u.path == '/log':
            n = int(urllib.parse.parse_qs(u.query).get('n', ['80'])[0])
            try:
                lines = open(f'{ROOT}/job.log', errors='replace').read().splitlines()[-n:]
            except FileNotFoundError:
                lines = []
            return self._send(200, '\n'.join(lines), 'text/plain')
        self._send(404, {'error': 'unknown'})

    def do_POST(self):
        if not self._ok(): return
        if self.path != '/run': return self._send(404, {'error': 'unknown'})
        pr = state['proc']
        if pr and pr.poll() is None: return self._send(409, {'error': 'job running'})
        script = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        with open(f'{ROOT}/job.sh', 'wb') as f: f.write(script)
        log = open(f'{ROOT}/job.log', 'ab')
        state['proc'] = subprocess.Popen(['bash', f'{ROOT}/job.sh'], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        self._send(200, {'started': True})

    def log_message(self, *a):
        pass


os.makedirs(ROOT, exist_ok=True)
http.server.ThreadingHTTPServer(('0.0.0.0', 8888), H).serve_forever()
