"""Train a Gaussian splat on a rented RunPod GPU, then delete the pod.

Usage:  python run_pod.py data.tgz out_dir
  data.tgz  = data/images/*.jpg + data/sparse/0/{cameras,images,points3D}.bin (COLMAP binary)
  needs RUNPOD_API_KEY in the environment (e.g. from the folder's .env file); never commit it.

What it does:
  1. creates a SECURE-cloud RTX 4090 pod (community pods had broken CUDA drivers)
     running pod_server.py, a tiny token-protected HTTP job server on port 8888
  2. uploads data.tgz, ckpt_to_web.py, ckpt_to_splat.py and job3.sh, runs the job
  3. polls the log, downloads results.tgz, deletes the pod (always, even on error)
Cost on the demo room: about 7.5 min of GPU time, roughly $0.45.
"""
import base64, json, os, secrets, sys, time, urllib.request

API = 'https://rest.runpod.io/v1'
KEY = os.environ['RUNPOD_API_KEY']
HERE = os.path.dirname(os.path.abspath(__file__))


def api(method, path, body=None):
    req = urllib.request.Request(API + path, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as r:
        t = r.read()
        return json.loads(t) if t else None


def job(base, token, method, path, data=None, timeout=600):
    req = urllib.request.Request(base + path, method=method, data=data, headers={'X-Token': token})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def main(data_tgz, out_dir):
    token = secrets.token_hex(24)
    srv = base64.b64encode(open(os.path.join(HERE, 'pod_server.py'), 'rb').read()).decode()
    pod = api('POST', '/pods', {
        'name': 'lexivo-splat', 'imageName': 'runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04',
        'gpuTypeIds': ['NVIDIA GeForce RTX 4090'], 'gpuCount': 1, 'cloudType': 'SECURE',
        'containerDiskInGb': 40, 'volumeInGb': 10, 'ports': ['8888/http'],
        'env': {'JOB_TOKEN': token, 'SRV': srv},
        'dockerStartCmd': ['bash', '-c', 'echo $SRV | base64 -d > /srv.py && python3 /srv.py']})
    pid = pod['id']
    base = f'https://{pid}-8888.proxy.runpod.net'
    print('pod', pid)
    try:
        for _ in range(120):  # wait for the job server
            try:
                if job(base, token, 'GET', '/health', timeout=10): break
            except Exception:
                time.sleep(5)
        for name, src in [('data.tgz', data_tgz), ('ckpt_to_web.py', 'ckpt_to_web.py'),
                          ('ckpt_to_splat.py', 'ckpt_to_splat.py')]:
            job(base, token, 'PUT', '/f/' + name, open(os.path.join(HERE, src) if src != data_tgz else src, 'rb').read())
        job(base, token, 'POST', '/run', open(os.path.join(HERE, 'job3.sh'), 'rb').read())
        while True:
            time.sleep(20)
            st = json.loads(job(base, token, 'GET', '/status'))
            print(job(base, token, 'GET', '/log?n=3').decode(errors='replace').strip()[-300:])
            if not st.get('running'):
                break
        os.makedirs(out_dir, exist_ok=True)
        open(os.path.join(out_dir, 'results.tgz'), 'wb').write(job(base, token, 'GET', '/f/results.tgz', timeout=1800))
        print('saved', os.path.join(out_dir, 'results.tgz'))
    finally:
        api('DELETE', f'/pods/{pid}')
        print('pod deleted')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
