"""Download the demo room photos (Mip-NeRF 360 "room" scene, Hugging Face copy) and resize them.

Usage:  python download_data.py [work_folder] [step]
  step 3 = every 3rd photo (104 images, used in the demo); step 1 = all 311.
"""
import json, sys, urllib.request
from pathlib import Path
from PIL import Image

work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / 'work'
step = int(sys.argv[2]) if len(sys.argv) > 2 else 3
raw, img = work / 'raw', work / 'img'
raw.mkdir(parents=True, exist_ok=True); img.mkdir(parents=True, exist_ok=True)

api = 'https://huggingface.co/api/datasets/nvs-bench/mipnerf360/tree/main/room/images'
files = sorted(d['path'] for d in json.load(urllib.request.urlopen(api)))[::step]
print(f'{len(files)} images')
for i, p in enumerate(files, 1):
    name = Path(p).name
    dst = raw / name
    if not dst.exists():
        urllib.request.urlretrieve(f'https://huggingface.co/datasets/nvs-bench/mipnerf360/resolve/main/{p}', dst)
    out = img / name.replace('.JPG', '.jpg')
    if not out.exists():
        im = Image.open(dst); w, h = im.size
        im.resize((1000, round(h * 1000 / w)), Image.LANCZOS).save(out, quality=92)
    print(f'\r{i}/{len(files)}', end='', flush=True)
print('\ndone:', img)
