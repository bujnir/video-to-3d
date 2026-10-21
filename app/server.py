"""Small local web app around pipeline.py: pick a video, paste a RunPod key, get every output.

Run:  python app/server.py      then open http://localhost:8080
Runs and their outputs are kept in DATA_DIR (default ./runs; /data inside Docker).
The RunPod key is handed to the pipeline process through its environment only. It is not
saved, logged or sent anywhere except RunPod's API.
"""
import html, json, os, re, shutil, signal, subprocess, sys, time, uuid
from pathlib import Path
from flask import Flask, abort, jsonify, redirect, request, send_from_directory

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get('DATA_DIR', ROOT / 'runs')).resolve(); DATA.mkdir(parents=True, exist_ok=True)
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 ** 3   # videos up to 8 GB
procs = {}

PAGE = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Video to 3D</title><style>
:root{--bg:#F5F6F7;--ink:#1B2329;--muted:#5D6B76;--rule:#D9DEE2;--acc:#E8642C;--card:#fff}
*{box-sizing:border-box}[hidden]{display:none!important}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
.wrap{max-width:760px;margin:0 auto;padding:32px 16px 60px}h1{font-size:1.8rem;margin:0 0 4px}h2{font-size:1.15rem;margin:28px 0 10px}
.sub{color:var(--muted);margin:0 0 24px}.card{background:var(--card);border:1px solid var(--rule);border-radius:8px;padding:20px}
label{display:block;font-weight:600;margin:14px 0 6px}label:first-child{margin-top:0}.hint{font-weight:400;color:var(--muted);font-size:.88rem}
input[type=text],input[type=password],select{width:100%;padding:10px 12px;border:1px solid var(--rule);border-radius:6px;font:inherit;background:#fff}
input[type=file]{font:inherit}.row{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media(max-width:560px){.row{grid-template-columns:1fr}}
button,.btn{display:inline-block;margin-top:18px;padding:11px 18px;border:0;border-radius:6px;background:var(--acc);color:#fff;font:600 1rem system-ui,sans-serif;cursor:pointer;text-decoration:none}
.btn.sec{background:#fff;color:var(--ink);border:1px solid var(--rule)}button[disabled]{opacity:.5}
ol.steps{padding-left:1.3em}ol.steps li{margin:4px 0;color:var(--muted)}ol.steps li.now{color:var(--ink);font-weight:600}ol.steps li.done{color:#2E7D4F}
pre{background:#11181E;color:#C9D3DC;padding:12px;border-radius:6px;font-size:.78rem;max-height:320px;overflow:auto;white-space:pre-wrap}
.err{background:#FDECEA;border:1px solid #F5C2BD;padding:12px;border-radius:6px;white-space:pre-wrap}
table{width:100%;border-collapse:collapse;font-size:.92rem}td{padding:6px 4px;border-top:1px solid var(--rule)}a{color:#B9481C}
</style></head><body><div class="wrap">%BODY%</div></body></html>'''


def page(body): return PAGE.replace('%BODY%', body)


def run_dir(rid):
    if not re.fullmatch(r'[0-9a-z\-]{6,60}', rid): abort(404)
    d = DATA / rid
    if not d.is_dir(): abort(404)
    return d


@app.get('/')
def index():
    has_env_key = bool(os.environ.get('RUNPOD_API_KEY'))
    rows = ''
    for d in sorted(DATA.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:20]:
        st = d / 'output' / 'status.json'
        s = json.load(open(st)) if st.exists() else {'state': 'starting'}
        meta = json.load(open(d / 'meta.json')) if (d / 'meta.json').exists() else {}
        rows += f'<tr><td><a href="/run/{d.name}">{html.escape(meta.get("name", d.name))}</a></td><td>{html.escape(meta.get("created", ""))}</td><td>{s["state"]}</td></tr>'
    key_hint = 'Optional: a key is already set for this app (from .env).' if has_env_key else 'Used for this run only. It is not saved.'
    return page(f'''<h1>Video to 3D</h1><p class="sub">Turn a walk-through video of a room into a photo-real 3D model, a smooth surface and a SketchUp-style shape model.</p>
<form class="card" method="post" action="/start" enctype="multipart/form-data" onsubmit="this.querySelector('button').disabled=true;this.querySelector('button').textContent='Uploading…'">
<label>Video <span class="hint">· .mp4 / .mov, or a .zip of photos. Walk slowly; 30–90 seconds is plenty for a room.</span></label>
<input type="file" name="video" accept="video/*,.zip" required>
<label>Name <span class="hint">· for the output folder</span></label><input type="text" name="name" placeholder="living-room" maxlength="40">
<label>RunPod API key <span class="hint">· {key_hint}</span></label><input type="password" name="key" autocomplete="off" {"" if has_env_key else "required"}>
<div class="row"><div><label>Quality</label><select name="steps"><option value="7000">Preview (fastest)</option><option value="15000" selected>Standard</option><option value="30000">Best (slowest)</option></select></div>
<div><label>Frames per second of video</label><select name="fps"><option value="1">1 (long or slow video)</option><option value="2" selected>2 (typical)</option><option value="3">3</option><option value="4">4 (short or fast video)</option></select></div></div>
<button type="submit">Start</button>
<p class="hint" style="margin:12px 0 0">Typical run: 15–30 minutes. The GPU step rents a RunPod GPU (about $0.90/hour, so roughly $0.20–0.60 per room) and always shuts it down at the end.</p>
</form>
<h2>Runs</h2><table>{rows or '<tr><td class="hint">No runs yet.</td></tr>'}</table>''')


@app.post('/start')
def start():
    f = request.files.get('video')
    if not f or not f.filename: abort(400, 'no file')
    key = (request.form.get('key') or '').strip() or os.environ.get('RUNPOD_API_KEY', '')
    if not key: abort(400, 'RunPod API key is required')
    name = re.sub(r'[^A-Za-z0-9_\-]+', '-', request.form.get('name') or Path(f.filename).stem).strip('-')[:40] or 'room'
    rid = time.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:4]
    d = DATA / rid; (d / 'input').mkdir(parents=True)
    ext = Path(f.filename).suffix.lower() or '.mp4'
    vid = d / 'input' / f'input{ext}'; f.save(vid)
    steps = request.form.get('steps', '15000'); fps = request.form.get('fps', '2')
    steps = steps if steps in ('7000', '15000', '30000') else '15000'; fps = fps if fps in ('1', '2', '3', '4') else '2'
    json.dump({'name': name, 'file': f.filename, 'created': time.strftime('%Y-%m-%d %H:%M'), 'steps': steps, 'fps': fps}, open(d / 'meta.json', 'w'))
    env = dict(os.environ, RUNPOD_API_KEY=key, PYTHONUNBUFFERED='1')
    log = open(d / 'run.log', 'w')
    launch(rid, env, log)
    return redirect(f'/run/{rid}')


def launch(rid, env, log, resume=False):
    d = DATA / rid; meta = json.load(open(d / 'meta.json'))
    vid = next((d / 'input').glob('input.*'))
    cmd = [sys.executable, str(ROOT / 'app' / 'pipeline.py'), str(vid), '--out', str(d / 'output'),
           '--name', meta['name'], '--steps', meta['steps'], '--fps', meta['fps']] + (['--resume'] if resume else [])
    procs[rid] = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


@app.post('/api/run/<rid>/retry')
def retry(rid):
    d = run_dir(rid); p = procs.get(rid)
    if p and p.poll() is None: return jsonify({'ok': False, 'error': 'still running'}), 409
    key = ((request.get_json(silent=True) or {}).get('key') or '').strip() or os.environ.get('RUNPOD_API_KEY', '')
    if not key: return jsonify({'ok': False, 'error': 'RunPod API key is required'}), 400
    log = open(d / 'run.log', 'a'); log.write('\n==== retry: continuing from the step that stopped ====\n'); log.flush()
    launch(rid, dict(os.environ, RUNPOD_API_KEY=key, PYTHONUNBUFFERED='1'), log, resume=True)
    return jsonify({'ok': True})


@app.get('/run/<rid>')
def run_page(rid):
    d = run_dir(rid); meta = json.load(open(d / 'meta.json'))
    return page(f'''<p><a href="/">← All runs</a></p><h1>{html.escape(meta["name"])}</h1>
<p class="sub">{html.escape(meta["file"])} · started {meta["created"]} · quality {meta["steps"]} steps · {meta["fps"]} frames/s</p>
<div class="card"><ol class="steps" id="steps"></ol><p id="msg" class="hint"></p><div id="err"></div>
<div id="done" hidden><a class="btn" href="/runs/{rid}/output/viewer/index.html" target="_blank">Open the 3D viewer</a>
<a class="btn sec" id="zip" href="#">Download all files (.zip)</a>
<p class="hint">Files are also in the output folder: runs/{rid}/output</p></div>
<div id="retrybox" hidden><label class="hint" for="rkey" id="rkeyl">RunPod API key for the retry</label> <input type="password" id="rkey" autocomplete="off" style="max-width:340px">
<button class="btn" id="retry" type="button">Retry from the failed step</button></div>
<button class="btn sec" id="cancel" type="button">Cancel run</button></div>
<h2>Log</h2><pre id="log"></pre>
<script>
const rid = {json.dumps(rid)};
async function tick(){{
  const r = await fetch('/api/run/' + rid).then(r => r.json()).catch(() => null);
  if (!r) return setTimeout(tick, 3000);
  const s = r.status || {{}}; const steps = s.steps || [];
  document.getElementById('steps').innerHTML = steps.map((t, i) => `<li class="${{i < s.step || s.state === 'done' ? 'done' : (i === s.step && s.state === 'running' ? 'now' : '')}}">${{t}}${{i < s.step || s.state === 'done' ? ' ✓' : ''}}</li>`).join('');
  document.getElementById('msg').textContent = s.state === 'running' ? `Working… ${{s.elapsed_min || 0}} min so far` : (s.state === 'done' ? `Finished in ${{s.elapsed_min}} min.` : (r.alive ? 'Starting…' : ''));
  const lg = document.getElementById('log'); const atEnd = lg.scrollTop + lg.clientHeight >= lg.scrollHeight - 20;
  lg.textContent = r.log; if (atEnd) lg.scrollTop = lg.scrollHeight;
  if (s.state === 'error') document.getElementById('err').innerHTML = '<p class="err"></p>', document.querySelector('#err .err').textContent = 'Stopped: ' + (s.error || 'unknown error');
  if (s.state === 'done') {{ document.getElementById('done').hidden = false; document.getElementById('zip').href = r.zip; }}
  document.getElementById('cancel').hidden = !r.alive;
  document.getElementById('retrybox').hidden = !(s.state === 'error' && !r.alive);
  if (r.has_env_key) {{ document.getElementById('rkey').hidden = true; document.getElementById('rkeyl').hidden = true; }}
  if (s.state !== 'error') document.getElementById('err').innerHTML = '';
  if (r.alive || !s.state || s.state === 'running') setTimeout(tick, 3000);
}}
document.getElementById('retry').onclick = async () => {{
  const res = await fetch('/api/run/' + rid + '/retry', {{method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify({{key: document.getElementById('rkey').value}})}}).then(r => r.json());
  if (!res.ok) return alert(res.error || 'Could not retry');
  document.getElementById('retrybox').hidden = true; setTimeout(tick, 1500);
}};
document.getElementById('cancel').onclick = async () => {{ if (confirm('Cancel this run? The GPU pod will be shut down.')) await fetch('/api/run/' + rid + '/cancel', {{method: 'POST'}}); }};
tick();
</script>''')


@app.get('/api/run/<rid>')
def run_api(rid):
    d = run_dir(rid); st = d / 'output' / 'status.json'
    p = procs.get(rid); alive = bool(p and p.poll() is None)
    status = json.load(open(st)) if st.exists() else {}
    if status.get('state') == 'running' and not alive:
        status = dict(status, state='error', error=status.get('error') or 'The pipeline process ended unexpectedly; see the log.')
    logtxt = (d / 'run.log').read_text(errors='replace')[-12000:] if (d / 'run.log').exists() else ''
    zips = list((d / 'output').glob('*.zip')); meta = json.load(open(d / 'meta.json'))
    return jsonify({'status': status, 'alive': alive, 'log': logtxt, 'zip': f'/runs/{rid}/output/{zips[0].name}' if zips else None, 'name': meta['name'],
                    'has_env_key': bool(os.environ.get('RUNPOD_API_KEY'))})


@app.post('/api/run/<rid>/cancel')
def cancel(rid):
    run_dir(rid); p = procs.get(rid)
    if p and p.poll() is None:
        os.killpg(p.pid, signal.SIGTERM)  # pipeline and run_pod catch this and delete the GPU pod
    return jsonify({'ok': True})


@app.get('/runs/<path:sub>')
def files(sub):
    if '_work' in sub.split('/'): abort(404)
    return send_from_directory(DATA, sub)


def shutdown(*_):
    # app stopping (e.g. docker stop): cancel running pipelines so their GPU pods are deleted
    live = [p for p in procs.values() if p.poll() is None]
    for p in live: os.killpg(p.pid, signal.SIGTERM)
    for p in live:
        try: p.wait(timeout=45)
        except subprocess.TimeoutExpired: pass
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, shutdown)
    print(f'Video to 3D: open http://localhost:{os.environ.get("PORT", "8080")}  (runs are saved in {DATA})', flush=True)
    app.run(host=os.environ.get('HOST', '0.0.0.0'), port=int(os.environ.get('PORT', 8080)), threaded=True)
