"""Train a Gaussian splat on a rented RunPod GPU, then delete the pod.

Usage:  python run_pod.py data.tgz out_dir [steps]
  data.tgz  from pack_for_gpu.py (photos + COLMAP model)
  steps     training steps, default 15000 (7000 = quicker preview)
Needs RUNPOD_API_KEY in the environment. The key is only sent to RunPod's API and never printed.

What it does:
  1. rents a SECURE-cloud GPU pod (RTX 4090 first, then other 24 GB+ cards if none are free;
     community-cloud pods had broken CUDA drivers in our tests)
  2. starts pod_server.py on it (a tiny token-protected job server) and uploads the data,
     ckpt_to_web.py, ckpt_to_splat.py and train_job.sh
  3. runs the job, prints progress, downloads results.tgz (includes room_full.splat.gz)
  4. always deletes the pod at the end, also after errors or Ctrl+C
Writes out_dir/results.tgz and out_dir/gpu_run.json (GPU, minutes, estimated cost).
"""
import base64, json, os, secrets, signal, sys, time, urllib.error, urllib.request

signal.signal(signal.SIGTERM, lambda *a: sys.exit(143))  # cancel -> still delete the pod (finally block)

API = 'https://rest.runpod.io/v1'
HERE = os.path.dirname(os.path.abspath(__file__))
UA = 'video-to-3d/1.0 (+https://github.com/bujnir/video-to-3d)'  # RunPod's firewall blocks Python's default user agent
GPUS = ['NVIDIA GeForce RTX 4090', 'NVIDIA RTX A6000', 'NVIDIA L40S', 'NVIDIA GeForce RTX 3090', 'NVIDIA RTX A5000']
IMAGE = 'runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04'


def api(method, path, body=None):
    req = urllib.request.Request(API + path, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={'Authorization': 'Bearer ' + os.environ['RUNPOD_API_KEY'], 'Content-Type': 'application/json', 'User-Agent': UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            t = r.read()
            return json.loads(t) if t else None
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors='replace')[:300]
        if e.code == 401 or (e.code == 403 and 'error code' not in msg):
            raise SystemExit('RunPod rejected the API key (check that it is correct and has read/write access).')
        raise RuntimeError(f'RunPod API {method} {path}: HTTP {e.code} {msg}')


def job(base, token, method, path, data=None, timeout=600, tries=4):
    for k in range(tries):  # the proxy in front of the pod gives short 404/502s while it settles
        req = urllib.request.Request(base + path, method=method, data=data, headers={'X-Token': token, 'User-Agent': UA})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            code = getattr(e, 'code', None)
            if k == tries - 1 or code in (400, 403, 409): raise
            time.sleep(10)


def create_pod(token):
    srv = base64.b64encode(open(os.path.join(HERE, 'pod_server.py'), 'rb').read()).decode()
    last = None
    for gpu in GPUS:  # one at a time, so we know which card we got
        try:
            pod = api('POST', '/pods', {
                'name': 'video-to-3d-splat', 'imageName': IMAGE, 'gpuTypeIds': [gpu], 'gpuCount': 1,
                'cloudType': 'SECURE', 'containerDiskInGb': 40, 'volumeInGb': 10, 'ports': ['8888/http'],
                'env': {'JOB_TOKEN': token, 'SRV': srv},
                'dockerStartCmd': ['bash', '-c', 'echo $SRV | base64 -d > /srv.py && python3 /srv.py']})
            return pod, gpu
        except RuntimeError as e:
            last = e; print(f'no {gpu} available, trying the next card', flush=True)
    raise RuntimeError(f'no GPU pod could be created: {last}')


def main(data_tgz, out_dir, steps=15000):
    if not os.environ.get('RUNPOD_API_KEY'):
        raise SystemExit('RUNPOD_API_KEY is not set.')
    os.makedirs(out_dir, exist_ok=True)
    token = secrets.token_hex(24)
    t0 = time.time()
    pod, gpu = create_pod(token)
    pid, rate = pod['id'], float(pod.get('costPerHr') or 0)
    base = f'https://{pid}-8888.proxy.runpod.net'
    print(f'pod {pid} on {gpu} (${rate:.2f}/h)', flush=True)
    ok = False
    try:
        for i in range(150):  # wait up to ~12 min for the pod to boot and *our* job server to answer
            try:  # RunPod's proxy can answer before the pod is up, so check for the server's own reply
                if json.loads(job(base, token, 'GET', '/health', timeout=10, tries=1)).get('ok') is True: break
            except Exception:
                pass
            if i % 6 == 0: print('waiting for the GPU pod to start…', flush=True)
            time.sleep(5)
        else:
            raise RuntimeError('the GPU pod did not start in time')
        print('uploading data', flush=True)
        job(base, token, 'PUT', '/f/data.tgz', open(data_tgz, 'rb').read(), timeout=1800)
        for name in ('ckpt_to_web.py', 'ckpt_to_splat.py'):
            job(base, token, 'PUT', '/f/' + name, open(os.path.join(HERE, name), 'rb').read())
        script = f'export STEPS={int(steps)}\n'.encode() + open(os.path.join(HERE, 'train_job.sh'), 'rb').read()
        job(base, token, 'POST', '/run', script)
        last = ''
        while True:
            time.sleep(15)
            st = json.loads(job(base, token, 'GET', '/status'))
            tail = job(base, token, 'GET', '/log?n=4').decode(errors='replace').strip()
            if tail and tail != last:
                print(tail.splitlines()[-1][-200:], flush=True); last = tail
            if not st.get('running'):
                break
        full = job(base, token, 'GET', '/log?n=60').decode(errors='replace')
        if st.get('exit') != 0 or 'ALL_DONE' not in full:
            raise RuntimeError('GPU job failed. Last log lines:\n' + full[-3000:])
        print('downloading results', flush=True)
        open(os.path.join(out_dir, 'results.tgz'), 'wb').write(job(base, token, 'GET', '/f/results.tgz', timeout=3600))
        ok = True
    finally:
        try:
            api('DELETE', f'/pods/{pid}'); print('pod deleted', flush=True)
        except Exception as e:
            print(f'WARNING: could not delete pod {pid}; delete it in the RunPod console. ({e})', flush=True)
        mins = (time.time() - t0) / 60
        json.dump({'gpu': gpu, 'pod_id': pid, 'usd_per_hour': rate, 'minutes': round(mins, 1),
                   'est_cost_usd': round(rate * mins / 60, 2), 'steps': int(steps), 'ok': ok},
                  open(os.path.join(out_dir, 'gpu_run.json'), 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 15000)
