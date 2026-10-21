"""Export the reconstruction to a compact JSON for the web viewer: points (xyz+rgb) and camera poses."""
import pycolmap, numpy as np, json
from pathlib import Path
import sys

# Work folder: first command-line argument, default ./work next to this script
root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / 'work'
rec = pycolmap.Reconstruction(root / 'sfm')
n_in = len(list((root / 'img').glob('*.jpg')))
pts = []
for pid, p in rec.points3D.items():
    if p.error > 2.0 or p.track.length() < 3:
        continue
    pts.append([*p.xyz, *p.color])
P = np.array(pts, dtype=np.float64)

cams = []
for iid, im in rec.images.items():
    if not im.has_pose:
        continue
    T = im.cam_from_world()
    R = T.rotation.matrix(); t = T.translation
    C = -R.T @ t                      # camera centre in world
    fwd = R.T @ np.array([0, 0, 1.0])  # viewing direction
    up = R.T @ np.array([0, -1.0, 0])
    cams.append({'name': im.name, 'c': C.tolist(), 'f': fwd.tolist(), 'u': up.tolist()})
cams.sort(key=lambda c: c['name'])

# normalise: centre on points' median, rotate so average camera "up" is +Y, scale to ~10 units
xyz = P[:, :3]
centre = np.median(xyz, 0)
upv = np.mean([c['u'] for c in cams], 0); upv /= np.linalg.norm(upv)
y = upv; x = np.cross(y, [0, 0, 1.0]); x /= np.linalg.norm(x); z = np.cross(x, y)
M = np.stack([x, y, z])
def tf(v, pos=True):
    v = np.asarray(v, float)
    return M @ ((v - centre) if pos else v)
xyz2 = (M @ (xyz - centre).T).T
# robust extent
lo, hi = np.percentile(xyz2, 2, 0), np.percentile(xyz2, 98, 0)
s = 10.0 / np.max(hi - lo)
keep = np.all((xyz2 > lo - (hi - lo) * 0.3) & (xyz2 < hi + (hi - lo) * 0.3), 1)
xyz2 = xyz2[keep] * s
rgb = P[keep, 3:].astype(int)
out = {
    'points': np.round(xyz2, 3).flatten().tolist(),
    'colors': rgb.flatten().tolist(),
    'cams': [{'name': c['name'], 'c': np.round(tf(c['c']) * s, 3).tolist(),
              'f': np.round(tf(c['f'], False), 3).tolist(),
              'u': np.round(tf(c['u'], False), 3).tolist()} for c in cams],
    'stats': {'images_in': n_in, 'registered': rec.num_reg_images(), 'points': int(keep.sum()),
              'points_raw': rec.num_points3D(),
              'reproj_err_px': round(float(np.mean([p.error for p in rec.points3D.values()])), 3)},
}
json.dump(out, open(Path(__file__).resolve().parent.parent / 'viewer' / 'scene.json', 'w'), separators=(',', ':'))
print(out['stats'], len(json.dumps(out)) / 1e3, 'KB')
