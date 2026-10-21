"""Turn a drone or phone video into sharp frames ready for sfm.py.

Usage:  python video_to_frames.py VIDEO.mp4 [work_folder] [frames_per_second] [width]
Defaults: 2 frames per second, 1000 px wide. Needs ffmpeg on the PATH.

Steps:
  1. ffmpeg pulls frames at the chosen rate and resizes them.
  2. Each frame gets a sharpness score (variance of the Laplacian);
     the blurriest 15% are dropped, since motion blur breaks feature matching.
"""
import subprocess, sys, shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

video = Path(sys.argv[1])
work = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent / 'work'
fps = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
width = int(sys.argv[4]) if len(sys.argv) > 4 else 1000

tmp, img = work / 'frames_all', work / 'img'
for d in (tmp, img):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)

subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', str(video),
                '-vf', f'fps={fps},scale={width}:-2', '-q:v', '2',
                str(tmp / 'frame_%05d.jpg')], check=True)
frames = sorted(tmp.glob('*.jpg'))

def sharpness(p):
    g = Image.open(p).convert('L').filter(ImageFilter.FIND_EDGES)
    return float(np.asarray(g, dtype=np.float32).var())

scores = {p: sharpness(p) for p in frames}
cut = np.percentile(list(scores.values()), 15) if len(frames) > 10 else -1
kept = [p for p in frames if scores[p] > cut]
for p in kept:
    shutil.copy(p, img / p.name)
shutil.rmtree(tmp)
print(f'{len(frames)} frames extracted, {len(kept)} sharp frames kept in {img}')
