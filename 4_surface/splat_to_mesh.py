"""Gaussian splat -> smooth, untextured surface mesh (geometry only).

Usage: python splat_to_mesh.py room_full.splat splat_view.json out_dir [voxel]

How it works:
  1. keep solid, small Gaussians (opaque surface blobs, not haze or floaters)
  2. for each of the photo cameras, render a depth map of those Gaussians (z-buffer)
  3. fuse all depth maps into one volume (TSDF fusion, Open3D) and extract the surface
  4. clean (keep large connected pieces), smooth (Taubin, keeps volume), simplify
Output: mesh.ply (full), mesh.glb (viewer-friendly), mesh_stats.json
The model is in the splat's normalised units, not metres, until a scale reference is added.
"""
import json, os, sys, time
import numpy as np
import open3d as o3d
from scipy.ndimage import median_filter

splat_path, view_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
voxel = float(sys.argv[4]) if len(sys.argv) > 4 else 0.006
os.makedirs(out, exist_ok=True)
t0 = time.time()

# ---- 1. load .splat (32 bytes: pos f32x3, scale f32x3, rgba u8x4, rot u8x4)
raw = np.fromfile(splat_path, dtype=np.uint8).reshape(-1, 32)
pos = raw[:, :12].copy().view(np.float32).reshape(-1, 3)
scl = raw[:, 12:24].copy().view(np.float32).reshape(-1, 3)
alpha = raw[:, 27] / 255.0
keep = (alpha > 0.25) & (scl.max(1) < float(os.environ.get('SMAX', 0.08)))
lo, hi = np.percentile(pos[keep], 0.5, 0), np.percentile(pos[keep], 99.5, 0)
keep &= np.all((pos > lo) & (pos < hi), 1)
P, S = pos[keep].astype(np.float64), scl[keep].max(1)
print(f'gaussians {len(pos):,} -> surface candidates {len(P):,}')

# ---- 2+3. depth maps per camera, fused into a TSDF volume
view = json.load(open(view_path))
DS = int(os.environ.get('DS', 2))  # depth maps at 1/DS of photo resolution
W, H = view['w'] // DS, view['h'] // DS
f = view['fx'] / DS
# volume grid around the surface candidates
gmin, gmax = P.min(0) - 3 * voxel, P.max(0) + 3 * voxel
dims = np.ceil((gmax - gmin) / voxel).astype(int)
print('grid', dims.tolist(), f'{dims.prod() / 1e6:.1f}M voxels')
axes = [gmin[i] + voxel * (np.arange(dims[i]) + 0.5) for i in range(3)]
X = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3).astype(np.float32)
trunc = voxel * 4
RMAX = int(os.environ.get('RMAX', 1))
from scipy.ndimage import maximum_filter, minimum_filter
tsdf = np.ones(len(X), np.float32)
wsum = np.zeros(len(X), np.float32)
cams = view['cams']
MAXV = int(os.environ.get('MAXVIEWS', 120))  # long videos: fuse an evenly spaced subset of views
if len(cams) > MAXV:
    cams = [cams[i] for i in np.linspace(0, len(cams) - 1, MAXV).round().astype(int)]
for ci, cam in enumerate(cams):
    R, t = np.array(cam['R']).reshape(3, 3), np.array(cam['t'])
    pc = P @ R.T + t
    z = pc[:, 2]
    ok = z > 0.05
    u = np.round(f * pc[ok, 0] / z[ok] + W / 2 - 0.5).astype(int)
    v = np.round(f * pc[ok, 1] / z[ok] + H / 2 - 0.5).astype(int)
    zz, rr = z[ok], np.clip(np.round(f * S[ok] / z[ok]), 0, RMAX).astype(int)
    depth = np.zeros((H, W), np.float32)
    # every blob covers a small square sized by its footprint; write all samples far-to-near
    U, V, Z = [], [], []
    for dy in range(-RMAX, RMAX + 1):
        for dx in range(-RMAX, RMAX + 1):
            m = rr >= max(abs(dx), abs(dy))
            U.append(u[m] + dx); V.append(v[m] + dy); Z.append(zz[m])
    U, V, Z = np.concatenate(U), np.concatenate(V), np.concatenate(Z)
    inb = (U >= 0) & (U < W) & (V >= 0) & (V < H)
    U, V, Z = U[inb], V[inb], Z[inb]
    o = np.argsort(-Z, kind='stable')
    depth[V[o], U[o]] = Z[o]  # later (nearer) writes win
    depth = median_filter(depth, 3)
    # drop depth edges (silhouettes): they smear foreground into background and cause drips
    dmax, dmin = maximum_filter(depth, 5), minimum_filter(np.where(depth > 0, depth, 99), 5)
    depth[(dmax - dmin) > float(os.environ.get('EDGE', 0.08)) * np.maximum(depth, 0.1)] = 0
    # TSDF update for voxels this camera sees
    xc = X @ R.T.astype(np.float32) + t.astype(np.float32)
    zc = xc[:, 2]
    front = zc > 0.05
    idx = np.nonzero(front)[0]
    uu = np.round(f * xc[idx, 0] / zc[idx] + W / 2 - 0.5).astype(np.int32)
    vv = np.round(f * xc[idx, 1] / zc[idx] + H / 2 - 0.5).astype(np.int32)
    inb = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
    idx, uu, vv = idx[inb], uu[inb], vv[inb]
    d = depth[vv, uu]
    sdf = d - zc[idx]
    good = (d > 0) & (sdf > -trunc)
    idx, sdf = idx[good], np.minimum(sdf[good] / trunc, 1.0)
    w = wsum[idx]
    tsdf[idx] = (tsdf[idx] * w + sdf) / (w + 1)
    wsum[idx] = w + 1
    if ci % 20 == 0: print(f'  view {ci}/{len(cams)}  {time.time() - t0:.0f}s', flush=True)
print(f'fused {len(cams)} depth maps in {time.time() - t0:.0f}s')
from skimage.measure import marching_cubes
vol = tsdf.reshape(dims)
mask = (wsum >= int(os.environ.get('MINW', 3))).reshape(dims)
vol[~mask] = 1.0
verts, faces, _, _ = marching_cubes(vol, 0.0, mask=mask)
verts = verts * voxel + gmin + voxel * 0.5
mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces[:, ::-1].copy()))

# ---- 4. surface, cleanup, smoothing, simplification
n_raw = len(mesh.triangles)
cl, n_tri, _ = mesh.cluster_connected_triangles()
cl, n_tri = np.asarray(cl), np.asarray(n_tri)
mesh.remove_triangles_by_mask(n_tri[cl] < 0.002 * n_raw)
mesh.remove_unreferenced_vertices()
mesh = mesh.filter_smooth_taubin(number_of_iterations=int(os.environ.get('SMOOTH', 40)))
target = 300_000
if len(mesh.triangles) > target:
    mesh = mesh.simplify_quadric_decimation(target)
mesh.remove_degenerate_triangles(); mesh.remove_unreferenced_vertices()
mesh.compute_vertex_normals()
o3d.io.write_triangle_mesh(f'{out}/mesh.ply', mesh)
stats = {'gaussians': int(len(pos)), 'surface_gaussians': int(len(P)), 'voxel': voxel,
         'triangles_raw': int(n_raw), 'triangles': int(len(mesh.triangles)),
         'vertices': int(len(mesh.vertices)), 'seconds': round(time.time() - t0)}
json.dump(stats, open(f'{out}/mesh_stats.json', 'w'))
print(stats)
