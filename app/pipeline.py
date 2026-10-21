"""Video to 3D in one command: every step, every output format.

Usage:
  python app/pipeline.py INPUT --out OUTPUT_DIR [--name NAME] [--fps 2] [--steps 15000] [--width 1000]
    INPUT   a video file (.mp4, .mov, ...), a folder of photos, or a .zip of photos
    --fps   frames taken per second of video (2 is right for a slow walk-through)
    --steps GPU training steps: 7000 = quick preview, 15000 = standard, 30000 = best
  The RunPod key is read from the RUNPOD_API_KEY environment variable and is only passed to
  the GPU step; it is never written to disk or to the logs.

Output folder:
  viewer/   index.html with the three interactive 3D views (open through a local web server)
  splat/    room.splat, room.ply (standard 3DGS), quality renders, fly-through video
  surface/  surface_topography.glb (height colours), surface.obj, surface_full.ply
  shapes/   shapes.dae (SketchUp), shapes.glb, shapes.obj, shapes_plan.png, shapes.json
  cameras/  colmap/ (COLMAP model), points.ply (sparse point cloud), cameras.json
  report.json, README.txt, logs/pipeline.log, and NAME.zip with all of the above
"""
import argparse, base64, glob, gzip, json, os, shutil, signal, subprocess, sys, tarfile, time, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
signal.signal(signal.SIGTERM, lambda *a: sys.exit(143))  # so clean-up (pod deletion) runs on cancel

ap = argparse.ArgumentParser()
ap.add_argument('input'); ap.add_argument('--out', required=True); ap.add_argument('--name')
ap.add_argument('--fps', type=float, default=2.0); ap.add_argument('--steps', type=int, default=15000)
ap.add_argument('--width', type=int, default=1000); ap.add_argument('--keep-work', action='store_true')
args = ap.parse_args()

src = Path(args.input).resolve()
out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
name = args.name or src.stem
work = out / '_work'; work.mkdir(exist_ok=True)
(out / 'logs').mkdir(exist_ok=True)
logf = open(out / 'logs' / 'pipeline.log', 'a', buffering=1)
STEPS = ['Extract frames', 'Solve camera positions', 'Package for the GPU', 'Train the 3D model on a RunPod GPU',
         'Build the smooth surface', 'Fit the shapes', 'Write all files']
t_start = time.time(); timings = {}


def status(i, msg, state='running', error=None):
    json.dump({'state': state, 'step': i, 'steps': STEPS, 'message': msg, 'error': error,
               'elapsed_min': round((time.time() - t_start) / 60, 1)}, open(out / 'status.json', 'w'))


def log(line):
    print(line, flush=True); logf.write(line + '\n')


def run(cmd, env_extra=None, keep_key=False):
    env = {k: v for k, v in os.environ.items() if keep_key or k != 'RUNPOD_API_KEY'}
    env.update(env_extra or {})
    p = subprocess.Popen([str(c) for c in cmd], cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = []
    for line in p.stdout:
        line = line.rstrip(); tail = (tail + [line])[-40:]; log('    ' + line)
    if p.wait() != 0:
        raise RuntimeError(f'{Path(str(cmd[1])).name} failed:\n' + '\n'.join(tail[-15:]))


def step(i, fn):
    status(i, STEPS[i]); log(f'[{i + 1}/{len(STEPS)}] {STEPS[i]}')
    t = time.time(); fn(); timings[STEPS[i]] = round((time.time() - t) / 60, 1)


def mb(p): return round(Path(p).stat().st_size / 1e6, 1)


# ---------------------------------------------------------------- steps
def frames():
    img = work / 'img'
    if src.is_dir() or src.suffix.lower() == '.zip':
        shutil.rmtree(img, ignore_errors=True); img.mkdir(parents=True)
        if src.is_dir():
            files = [p for p in sorted(src.iterdir()) if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
            for p in files: shutil.copy(p, img / p.name)
        else:
            with zipfile.ZipFile(src) as z:
                for n in z.namelist():
                    if n.lower().endswith(('.jpg', '.jpeg', '.png')) and not n.startswith('__MACOSX'):
                        (img / Path(n).name).write_bytes(z.read(n))
        # same size for all photos keeps COLMAP and the GPU step simple
        from PIL import Image
        for p in img.iterdir():
            im = Image.open(p).convert('RGB')
            if im.width > args.width: im = im.resize((args.width, round(im.height * args.width / im.width)), Image.LANCZOS)
            im.save(p.with_suffix('.jpg'), quality=95)
            if p.suffix.lower() != '.jpg': p.unlink()
    else:
        run([PY, '1_frames/video_to_frames.py', src, work, args.fps, args.width])
    n = len(list(img.glob('*.jpg')))
    log(f'    {n} frames ready')
    if n < 20:
        raise RuntimeError(f'Only {n} usable frames. Use a longer video or a higher --fps (frames per second).')


def poses():
    n = len(list((work / 'img').glob('*.jpg')))
    run([PY, '2_camera_poses/sfm.py', work, 'exhaustive' if n <= 150 else 'sequential'])
    lg = json.load(open(work / 'sfm' / 'log.json'))
    log(f"    {lg['reg_images']} of {n} frames placed, {lg['points3D']:,} 3D points")
    if lg['reg_images'] < max(10, 0.4 * n):
        raise RuntimeError(f"COLMAP could only place {lg['reg_images']} of {n} frames. Usual causes: motion blur, too fast "
                           'movement, plain walls, or spinning in place. Try a slower video with more overlap.')


def pack():
    run([PY, '3_gaussian_splat/pack_for_gpu.py', work, work / 'gpu_in'])


def gpu():
    if not os.environ.get('RUNPOD_API_KEY'):
        raise RuntimeError('No RunPod API key was given.')
    run([PY, '3_gaussian_splat/run_pod.py', work / 'gpu_in' / 'data.tgz', work / 'gpu_out', args.steps], keep_key=True)
    res = work / 'gpu_out' / 'res'; shutil.rmtree(res, ignore_errors=True); res.mkdir()
    with tarfile.open(work / 'gpu_out' / 'results.tgz') as t: t.extractall(res)
    with gzip.open(res / 'room_full.splat.gz', 'rb') as f, open(res / 'room.splat', 'wb') as g: shutil.copyfileobj(f, g)


def surface():
    view = work / 'gpu_in' / 'splat_view.json'
    run([PY, '4_surface/splat_to_mesh.py', work / 'gpu_out' / 'res' / 'room.splat', view, work / 'mesh', 0.011],
        env_extra={'DS': '2', 'RMAX': '2', 'MINW': '2', 'SMAX': '0.08', 'EDGE': '0.08', 'SMOOTH': '40', 'MAXVIEWS': '120'})
    (work / 'web').mkdir(exist_ok=True)
    run([PY, '4_surface/mesh_to_web.py', work / 'mesh' / 'mesh.ply', work / 'web', 220000, view])


def shapes():
    view = work / 'gpu_in' / 'splat_view.json'
    run([PY, '5_shapes/blockout.py', work / 'web' / 'room_mesh.b64.txt', view, work / 'shapes'])
    run([PY, '4_surface/export_desktop.py', work / 'web', view, work / 'shapes' / 'room_frame.json'])


def package():
    res, gi, sh, web = work / 'gpu_out' / 'res', work / 'gpu_in', work / 'shapes', work / 'web'
    d = {k: out / k for k in ('viewer', 'splat', 'surface', 'shapes', 'cameras')}
    for p in d.values(): p.mkdir(exist_ok=True)
    # splat
    shutil.copy(res / 'room.splat', d['splat'] / 'room.splat')
    run([PY, '3_gaussian_splat/splat_to_ply.py', d['splat'] / 'room.splat', d['splat'] / 'room.ply'])
    for v in glob.glob(str(res / 'videos' / '*.mp4'))[:1]: shutil.copy(v, d['splat'] / 'flythrough.mp4')
    (d['splat'] / 'quality_renders').mkdir(exist_ok=True)
    for r in sorted(glob.glob(str(res / 'renders' / '*.png'))): shutil.copy(r, d['splat'] / 'quality_renders' / Path(r).name)
    # surface
    shutil.copy(web / 'room_surface_topography.glb', d['surface'] / 'surface_topography.glb')
    shutil.copy(web / 'room_surface.obj', d['surface'] / 'surface.obj')
    shutil.copy(work / 'mesh' / 'mesh.ply', d['surface'] / 'surface_full.ply')
    # shapes
    for ext in ('dae', 'glb', 'obj'): shutil.copy(sh / f'room_blocks.{ext}', d['shapes'] / f'shapes.{ext}')
    shutil.copy(sh / 'blocks_top.png', d['shapes'] / 'shapes_plan.png'); shutil.copy(sh / 'blocks.json', d['shapes'] / 'shapes.json')
    # cameras
    shutil.rmtree(d['cameras'] / 'colmap', ignore_errors=True); (d['cameras'] / 'colmap').mkdir()
    with tarfile.open(gi / 'data.tgz') as t:
        for m in t.getmembers():
            if m.name.startswith('data/sparse/0/'):
                (d['cameras'] / 'colmap' / Path(m.name).name).write_bytes(t.extractfile(m).read())
    shutil.copy(gi / 'points.ply', d['cameras'] / 'points.ply'); shutil.copy(gi / 'splat_view.json', d['cameras'] / 'cameras.json')

    # viewer data
    v = d['viewer']
    shutil.copy(ROOT / 'app' / 'viewer_template.html', v / 'index.html')
    for k in (0, 1):
        (v / f'room_part{k}.b64.txt').write_text(base64.b64encode((res / 'web' / f'room_part{k}.bin').read_bytes()).decode())
    shutil.copy(gi / 'splat_view.json', v / 'splat_view.json'); shutil.copy(sh / 'blocks.json', v / 'blocks.json')
    shutil.copy(web / 'room_mesh.b64.txt', v / 'room_mesh.b64.txt')
    from PIL import Image
    first = sorted(glob.glob(str(res / 'renders' / '*.png')))
    if first:
        im = Image.open(first[0]).convert('RGB'); im.thumbnail((1400, 1400)); im.save(v / 'compare.jpg', quality=88)
    val = json.load(open(sorted(glob.glob(str(res / 'stats' / 'val_step*.json')))[-1]))
    conv = json.load(open(res / 'web' / 'web_convert.json'))
    run_info = json.load(open(work / 'gpu_out' / 'gpu_run.json'))
    mesh_info = json.load(open(work / 'mesh' / 'mesh_stats.json')); web_info = json.load(open(web / 'room_mesh_web.json'))
    sfm = json.load(open(work / 'sfm' / 'log.json')); blk = json.load(open(sh / 'blocks_stats.json'))
    tsec = float(open(res / 'train_seconds.txt').read().split()[-1]) if (res / 'train_seconds.txt').exists() else 0
    json.dump({'splats_total': conv['splats_total'], 'splats_shown': conv['splats_shown'], 'psnr': round(val['psnr'], 1),
               'ssim': round(val['ssim'], 3), 'lpips': round(val['lpips'], 3), 'train_min': round(tsec / 60),
               'cost': f"${run_info['est_cost_usd']}", 'mb': round(sum(conv['part_bytes']) / 1e6, 1), 'gpu': run_info['gpu']},
              open(v / 'splat_stats.json', 'w'))
    json.dump({'triangles': web_info['triangles'], 'vertices': web_info['vertices'], 'minutes': timings.get(STEPS[4])}, open(v / 'mesh_stats.json', 'w'))

    stats = {'frames_used': sfm.get('images'), 'cameras_solved': sfm['reg_images'], 'points3D': sfm['points3D'],
             'splats': conv['splats_total'], 'psnr': round(val['psnr'], 1), 'ssim': round(val['ssim'], 3),
             'surface_triangles': web_info['triangles'], 'shape_boxes': blk['boxes'], 'shape_objects': blk['objects']}
    groups = [('Photo-real model (Gaussian splat)', 'splat', {'room.splat': 'open at superspl.at/editor or other splat viewers',
              'room.ply': 'standard 3D Gaussian Splatting file (SuperSplat, Postshot, Blender add-ons)',
              'flythrough.mp4': 'video rendered from the model'}),
              ('Smooth surface', 'surface', {'surface_topography.glb': 'mesh with height colours (Blender, Windows 3D Viewer)',
               'surface.obj': 'plain mesh, same coordinates as the shapes', 'surface_full.ply': 'full-detail mesh (MeshLab, CloudCompare)'}),
              ('Shapes', 'shapes', {'shapes.dae': 'SketchUp: File > Import', 'shapes.glb': 'Blender, web viewers', 'shapes.obj': 'most 3D apps',
               'shapes_plan.png': 'plan view of the boxes', 'shapes.json': 'box list (room coordinates)'}),
              ('Cameras', 'cameras', {'points.ply': 'sparse 3D point cloud', 'cameras.json': 'camera positions used by the viewer'})]
    files = [{'group': g, 'items': [{'path': f'{folder}/{f}', 'what': w, 'mb': mb(out / folder / f)} for f, w in items.items() if (out / folder / f).exists()]}
             for g, folder, items in groups]
    report = {'name': name, 'input': src.name, 'created': time.strftime('%Y-%m-%d %H:%M'), 'settings': {'fps': args.fps, 'steps': args.steps, 'width': args.width},
              'stats': stats, 'gpu': run_info, 'minutes_per_step': timings, 'files': files}
    json.dump(report, open(out / 'report.json', 'w'), indent=1); shutil.copy(out / 'report.json', v / 'report.json')

    lines = [f'Video to 3D results: {name}', '=' * 40, '',
             'Open the 3D viewer: run  python -m http.server 8000  inside the viewer folder, then open http://localhost:8000',
             '(the app does this for you; browsers block 3D data opened straight from disk)', '']
    for g in files:
        lines.append(g['group']);
        lines += [f"  {it['path']:<34} {it['mb']:>7} MB  {it['what']}" for it in g['items']]; lines.append('')
    lines += ['cameras/colmap/  COLMAP model (cameras.bin, images.bin, points3D.bin)',
              'splat/quality_renders/  real frame | model render, for frames held out of training', '',
              'All models are in relative units. Add one known distance, IMU data or a depth sensor for real measurements.']
    (out / 'README.txt').write_text('\n'.join(lines) + '\n')
    zp = out / f'{name}.zip'
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and '_work' not in p.parts and p != zp and p.name != 'status.json':
                z.write(p, Path(name) / p.relative_to(out))
    log(f'    zip: {zp.name} ({mb(zp)} MB)')


# ---------------------------------------------------------------- run
try:
    for i, fn in enumerate([frames, poses, pack, gpu, surface, shapes, package]):
        step(i, fn)
    status(len(STEPS), 'Done', state='done')
    log(f'Done in {round((time.time() - t_start) / 60, 1)} min. Results: {out}')
    if not args.keep_work: shutil.rmtree(work, ignore_errors=True)
except BaseException as e:
    msg = str(e) or type(e).__name__
    cur = json.load(open(out / 'status.json')).get('step', 0) if (out / 'status.json').exists() else 0
    status(cur, 'Stopped' if isinstance(e, (KeyboardInterrupt, SystemExit)) else 'Failed', state='error', error=msg)
    log(f'ERROR: {msg}')
    sys.exit(1)
