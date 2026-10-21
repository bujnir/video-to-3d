"""Make step images: sample frames strip, SIFT keypoints on one frame, matches between two frames."""
import pycolmap, numpy as np, os, sys
from PIL import Image, ImageDraw
from pathlib import Path

# Work folder: first command-line argument, default ./work next to this script
root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / 'work'
img_dir = root / 'img'
out = Path(__file__).resolve().parent.parent / 'viewer'
out.mkdir(exist_ok=True)
names = sorted(os.listdir(img_dir))

# 1. frame strip: 6 evenly spaced frames
pick = [names[i] for i in np.linspace(0, len(names) - 1, 6).astype(int)]
th = [Image.open(img_dir / n).resize((300, 200)) for n in pick]
strip = Image.new('RGB', (300 * 3 + 8 * 2, 200 * 2 + 8), 'white')
for i, t in enumerate(th):
    strip.paste(t, ((i % 3) * 308, (i // 3) * 208))
strip.save(out / 'frames.jpg', quality=85)

# 2/3. SIFT on two neighbouring frames, then match with ratio test
a, b = names[20], names[23]
def sift(path):
    im = np.asarray(Image.open(path).convert('L')).astype(np.float32) / 255.0
    s = pycolmap.Sift(options=pycolmap.FeatureExtractionOptions())
    kp, desc = s.extract(im)
    return kp, desc
ka, da = sift(img_dir / a)
kb, db = sift(img_dir / b)
print('keypoints', len(ka), len(kb))

ia = Image.open(img_dir / a).convert('RGB')
d = ImageDraw.Draw(ia)
for x, y, *_ in ka[:2500]:
    d.ellipse([x - 2.5, y - 2.5, x + 2.5, y + 2.5], outline=(255, 200, 0), width=1)
ia.save(out / 'features.jpg', quality=85)

# brute-force match on float descriptors (L2), Lowe ratio + mutual check
da = da.astype(np.float32); db = db.astype(np.float32)
D = (da ** 2).sum(1)[:, None] + (db ** 2).sum(1)[None] - 2 * da @ db.T
nn = np.argsort(D, 1)[:, :2]
ratio = np.sqrt(np.maximum(D[np.arange(len(da)), nn[:, 0]], 0)) / np.sqrt(np.maximum(D[np.arange(len(da)), nn[:, 1]], 1e-9))
back = np.argmin(D, 0)
good = [(i, nn[i, 0]) for i in range(len(da)) if ratio[i] < 0.75 and back[nn[i, 0]] == i]
print('matches', len(good))
A = Image.open(img_dir / a).convert('RGB'); B = Image.open(img_dir / b).convert('RGB')
W, H = A.size
canvas = Image.new('RGB', (W * 2 + 10, H), 'white')
canvas.paste(A, (0, 0)); canvas.paste(B, (W + 10, 0))
d = ImageDraw.Draw(canvas)
rng = np.random.default_rng(0)
sel = rng.choice(len(good), size=min(140, len(good)), replace=False)
for k in sel:
    i, j = good[k]
    xa, ya = ka[i][:2]; xb, yb = kb[j][:2]
    c = tuple(int(v) for v in rng.integers(60, 255, 3))
    d.line([xa, ya, xb + W + 10, yb], fill=c, width=1)
    d.ellipse([xa - 3, ya - 3, xa + 3, ya + 3], outline=c, width=2)
    d.ellipse([xb + W + 7, yb - 3, xb + W + 13, yb + 3], outline=c, width=2)
canvas = canvas.resize((1400, int(H * 1400 / (W * 2 + 10))), Image.LANCZOS)
canvas.save(out / 'matches.jpg', quality=85)
import json
json.dump({'pair': [a, b], 'kp': [len(ka), len(kb)], 'matches': len(good)}, open(out / 'feat_stats.json', 'w'))
