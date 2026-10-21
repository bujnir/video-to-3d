"""Surface mesh -> SketchUp-style block model ("push/pull" extrusions).

Usage: python blockout.py room_mesh.b64.txt splat_view.json out_dir
Steps:
  1. room frame: floor plane by RANSAC, wall direction from vertical-surface normals
  2. top-down height map of the room (0.03 model-unit cells), small gaps filled
  3. heights snapped to 5 cm levels, speckles removed, each flat region becomes one
     outline polygon (simplified, so stair-steps turn into straight edges)
  4. every region is extruded from the floor to its height; tall regions become walls
     with one common height; the whole footprint gets a floor slab
Outputs: blocks.json (for the web viewer, with the transform back to splat space),
         room_blocks.glb / .dae / .obj (Y up, floor at 0), blocks_top.png (check image)
"""
import base64, json, os, struct, sys
import numpy as np
import open3d as o3d
import trimesh
from scipy import ndimage

mesh_b64, view_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
CELL, STEP = 0.03, 0.05
view = json.load(open(view_path))

# ---- load web mesh and sample points with normals
blob = base64.b64decode(open(mesh_b64).read())
nv, nf = struct.unpack('<II', blob[:8])
lo, hi = np.array(struct.unpack('<3f', blob[8:20])), np.array(struct.unpack('<3f', blob[20:32]))
V = lo + (hi - lo) * np.frombuffer(blob, '<u2', nv * 3, 32).reshape(-1, 3) / 65535
F = np.frombuffer(blob, '<u4', nf * 3, 32 + nv * 6).reshape(-1, 3).astype(np.int32)
m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
m.compute_vertex_normals()
pc = m.sample_points_uniformly(800_000)
P, N = np.asarray(pc.points), np.asarray(pc.normals)

# ---- 1. room frame
up = np.array(view['up']); up /= np.linalg.norm(up)
h0 = P @ up
cand = (np.abs(N @ up) > 0.9) & (h0 < np.percentile(h0, 30))
plane, _ = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[cand])).segment_plane(0.01, 3, 3000)
n = np.array(plane[:3]); s = np.linalg.norm(n); n /= s; d = plane[3] / s
if n @ up < 0: n, d = -n, -d
a0 = np.cross(n, [1, 0, 0]); a0 /= np.linalg.norm(a0); b0 = np.cross(n, a0)
wallish = np.abs(N @ n) < 0.2
ang = np.mod(np.arctan2(N[wallish] @ b0, N[wallish] @ a0), np.pi / 2)
hist, edges = np.histogram(ang, 90, (0, np.pi / 2))
hist = np.convolve(np.r_[hist[-2:], hist, hist[:2]], np.ones(5) / 5, 'valid')  # circular smoothing
th = edges[np.argmax(hist)] + (edges[1] - edges[0]) / 2
ax = np.cos(th) * a0 + np.sin(th) * b0; az = np.cross(ax, n)
X, Y, Z = P @ ax, P @ n + d, P @ az   # room frame: Y = height above floor

# ---- 2. height map
ytop = np.percentile(Y, 99.5)
x0, x1 = np.percentile(X, [0.3, 99.7]); z0, z1 = np.percentile(Z, [0.3, 99.7])
W, H = int(np.ceil((x1 - x0) / CELL)), int(np.ceil((z1 - z0) / CELL))
xi, zi = ((X - x0) / CELL).astype(int), ((Z - z0) / CELL).astype(int)
ok = (xi >= 0) & (xi < W) & (zi >= 0) & (zi < H) & (Y < ytop)
xi, zi, Yk = xi[ok], zi[ok], Y[ok]
cnt = np.zeros((H, W)); np.add.at(cnt, (zi, xi), 1)
above = Yk > 0.025
Hm = np.full((H, W), -1.0); np.maximum.at(Hm, (zi[above], xi[above]), Yk[above])
Hm[(Hm < 0) & (cnt > 0)] = 0.0                     # only floor seen here
Hm[cnt < 3] = np.nan                               # too little data
def nanmed(v):
    v = v[~np.isnan(v)]
    return np.median(v) if len(v) >= 5 else np.nan
Hs = ndimage.generic_filter(Hm, nanmed, size=5, mode='constant', cval=np.nan)

# ---- 3. room shell: rectangle around the wall-height surfaces
hi_pts = Y > 0.55 * ytop
rx0, rx1 = np.percentile(X[hi_pts], [1, 99]); rz0, rz1 = np.percentile(Z[hi_pts], [1, 99])
near_edge = hi_pts & ((np.minimum(X - rx0, rx1 - X) < 0.08) | (np.minimum(Z - rz0, rz1 - Z) < 0.08))
wall_h = float(np.percentile(Y[near_edge], 90))
T = 0.04  # wall thickness (model units)
shapes = [dict(kind='floor', x0=rx0 - T, x1=rx1 + T, z0=rz0 - T, z1=rz1 + T, y0=-0.03, y1=0.0)]
shapes += [dict(kind='wall', x0=rx0 - T, x1=rx1 + T, z0=rz0 - T, z1=rz0, y0=0, y1=wall_h, side='z0'),
           dict(kind='wall', x0=rx0 - T, x1=rx1 + T, z0=rz1, z1=rz1 + T, y0=0, y1=wall_h, side='z1'),
           dict(kind='wall', x0=rx0 - T, x1=rx0, z0=rz0, z1=rz1, y0=0, y1=wall_h, side='x0'),
           dict(kind='wall', x0=rx1, x1=rx1 + T, z0=rz0, z1=rz1, y0=0, y1=wall_h, side='x1')]

# ---- 4. furniture: objects in the height map -> a few axis-aligned boxes each
gx = x0 + (np.arange(W) + 0.5) * CELL; gz = z0 + (np.arange(H) + 0.5) * CELL
inside = (gz[:, None] > rz0 + 0.03) & (gz[:, None] < rz1 - 0.03) & (gx[None] > rx0 + 0.03) & (gx[None] < rx1 - 0.03)
floor_seen = np.zeros((H, W), bool); low = Yk < 0.025; floor_seen[zi[low], xi[low]] = True
Hf = np.nan_to_num(Hs, nan=0.0)
obj = inside & (Hf > 0.07) & (Hf < 0.8 * wall_h)
obj = ndimage.binary_opening(ndimage.binary_closing(obj, iterations=1), iterations=1)
lab, k = ndimage.label(obj)

def largest_rect(mask):
    """largest axis-aligned all-True rectangle in a boolean grid -> (area, r0, r1, c0, c1)"""
    best = (0, 0, 0, 0, 0); hgt = np.zeros(mask.shape[1], int)
    for r in range(mask.shape[0]):
        hgt = np.where(mask[r], hgt + 1, 0); st = []
        for c in range(len(hgt) + 1):
            hc = hgt[c] if c < len(hgt) else 0; start = c
            while st and st[-1][1] >= hc:
                start, hh = st.pop()
                if hh * (c - start) > best[0]: best = (hh * (c - start), r - hh + 1, r + 1, start, c)
            st.append((start, hc))
    return best

n_obj = 0
for j in range(1, k + 1):
    comp = lab == j
    if comp.sum() < 12: continue
    hv = Hf[comp]
    # split tall and low parts (e.g. sofa back vs seat) when the heights are clearly two groups
    tiers = [comp]
    if hv.max() - hv.min() > 0.12:
        cut = np.percentile(hv, 50)
        hi_m, lo_m = comp & (Hf > cut + 0.04), comp
        if hi_m.sum() >= 8: tiers = [lo_m, ndimage.binary_opening(hi_m)]
    n_obj += 1
    for ti, tm in enumerate(tiers):
        rem = tm.copy(); total = tm.sum()
        for _ in range(4):
            area, r0, r1, c0, c1 = largest_rect(rem)
            if area < 6 or (tm.sum() - rem.sum()) > 0.88 * total: break
            rem[r0:r1, c0:c1] = False
            cells = Hf[r0:r1, c0:c1][tm[r0:r1, c0:c1]]
            top = round(float(np.percentile(cells, 75 if ti == 0 else 85)) / 0.02) * 0.02
            bx = dict(kind='block', obj=n_obj, x0=x0 + c0 * CELL, x1=x0 + c1 * CELL, z0=z0 + r0 * CELL, z1=z0 + r1 * CELL, y0=0.0, y1=top)
            # open underneath (floor seen below most of it) and raised -> table: top slab + 4 legs
            if top > 0.15 and floor_seen[r0:r1, c0:c1].mean() > 0.35 and ti == 0:
                th_ = min(0.035, top * 0.15); lg = min(0.03, (bx['x1'] - bx['x0']) / 5, (bx['z1'] - bx['z0']) / 5)
                shapes.append(dict(bx, kind='table', y0=top - th_))
                for cx, cz in ((bx['x0'], bx['z0']), (bx['x1'] - lg, bx['z0']), (bx['x0'], bx['z1'] - lg), (bx['x1'] - lg, bx['z1'] - lg)):
                    shapes.append(dict(kind='leg', obj=n_obj, x0=cx, x1=cx + lg, z0=cz, z1=cz + lg, y0=0.0, y1=top - th_))
            else:
                shapes.append(bx)
print(f'room {rx1 - rx0:.2f} x {rz1 - rz0:.2f}, wall height {wall_h:.2f}; {n_obj} objects, '
      + ', '.join(f"{t} {sum(s['kind'] == t for s in shapes)}" for t in ('wall', 'block', 'table', 'leg')))

# ---- outputs
r4 = lambda v: round(float(v), 4)
web = {'note': 'room frame: x along a wall, y up from the floor, z across; splat space = x*ax + (y - d)*n + z*az',
       'ax': ax.round(6).tolist(), 'n': n.round(6).tolist(), 'az': az.round(6).tolist(), 'd': r4(d),
       'room': [r4(rx0), r4(rx1), r4(rz0), r4(rz1)], 'wall_h': r4(wall_h),
       'boxes': [{**{k2: (r4(v) if isinstance(v, float) else v) for k2, v in sh.items()}} for sh in shapes]}
json.dump(web, open(f'{out}/blocks.json', 'w'), separators=(',', ':'))
json.dump({'ax': web['ax'], 'n': web['n'], 'az': web['az'], 'd': web['d']}, open(f'{out}/room_frame.json', 'w'))

colours = {'floor': [205, 199, 188, 255], 'wall': [242, 241, 236, 255], 'block': [226, 212, 186, 255],
           'table': [196, 170, 132, 255], 'leg': [120, 110, 100, 255]}
scene = trimesh.Scene()
for i, sh in enumerate(shapes):
    ext = [sh['x1'] - sh['x0'], sh['y1'] - sh['y0'], sh['z1'] - sh['z0']]
    bxm = trimesh.creation.box(extents=ext)
    bxm.apply_translation([(sh['x0'] + sh['x1']) / 2, (sh['y0'] + sh['y1']) / 2, (sh['z0'] + sh['z1']) / 2])
    bxm.visual = trimesh.visual.ColorVisuals(bxm, face_colors=colours[sh['kind']])
    scene.add_geometry(bxm, geom_name=f"{sh['kind']}_{i:03d}")
scene.export(f'{out}/room_blocks.glb')
scene.export(f'{out}/room_blocks.obj')
from trimesh.exchange.dae import export_collada
open(f'{out}/room_blocks.dae', 'wb').write(export_collada([g for g in scene.dump()]))

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, a = plt.subplots(figsize=(14, 6))
a.scatter(X[::25], Z[::25], s=0.2, c='#bbb')
for sh in sorted(shapes, key=lambda s: s['y1']):
    c = {'floor': '#e8e4dc', 'wall': '#555'}.get(sh['kind'], plt.cm.viridis(min(sh['y1'] / wall_h, 1)))
    a.add_patch(plt.Rectangle((sh['x0'], sh['z0']), sh['x1'] - sh['x0'], sh['z1'] - sh['z0'], fc=c, ec='k', lw=0.6, alpha=0.9 if sh['kind'] != 'floor' else 0.4))
    if sh['kind'] in ('block', 'table'): a.text(sh['x0'], sh['z0'], f"{sh['y1']:.2f}", fontsize=6)
a.set_aspect('equal'); a.autoscale(); a.set_title('block model, plan view (colour = height)'); plt.tight_layout(); plt.savefig(f'{out}/blocks_top.png', dpi=90)
stats = {'objects': n_obj, 'boxes': len(shapes), 'room': web['room'], 'wall_h': web['wall_h'], 'floor_tilt_deg': round(float(np.degrees(np.arccos(n @ up))), 2)}
json.dump(stats, open(f'{out}/blocks_stats.json', 'w')); print(stats)
