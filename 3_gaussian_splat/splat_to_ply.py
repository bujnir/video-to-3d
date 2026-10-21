"""Convert a .splat file into the standard 3D Gaussian Splatting .ply layout.

Usage: python splat_to_ply.py room.splat room.ply
The .ply opens in SuperSplat, Postshot, Blender add-ons and most splat viewers. It carries the
base colour only (no view-dependent shading), which is what the .splat format stores.
"""
import sys
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
raw = np.fromfile(src, np.uint8).reshape(-1, 32)
pos = raw[:, :12].copy().view(np.float32).reshape(-1, 3)
scl = raw[:, 12:24].copy().view(np.float32).reshape(-1, 3)
rgba = raw[:, 24:28].astype(np.float32) / 255
rot = (raw[:, 28:32].astype(np.float32) - 128) / 128
rot /= np.maximum(np.linalg.norm(rot, axis=1, keepdims=True), 1e-8)
C0 = 0.28209479177387814
a = np.clip(rgba[:, 3], 1e-4, 1 - 1e-4)
cols = {'x': pos[:, 0], 'y': pos[:, 1], 'z': pos[:, 2], 'nx': 0 * pos[:, 0], 'ny': 0 * pos[:, 0], 'nz': 0 * pos[:, 0],
        'f_dc_0': (rgba[:, 0] - 0.5) / C0, 'f_dc_1': (rgba[:, 1] - 0.5) / C0, 'f_dc_2': (rgba[:, 2] - 0.5) / C0,
        'opacity': np.log(a / (1 - a)),
        'scale_0': np.log(np.maximum(scl[:, 0], 1e-8)), 'scale_1': np.log(np.maximum(scl[:, 1], 1e-8)), 'scale_2': np.log(np.maximum(scl[:, 2], 1e-8)),
        'rot_0': rot[:, 0], 'rot_1': rot[:, 1], 'rot_2': rot[:, 2], 'rot_3': rot[:, 3]}
arr = np.zeros(len(pos), dtype=[(k, '<f4') for k in cols])
for k, v in cols.items():
    arr[k] = v
with open(dst, 'wb') as f:
    f.write(('ply\nformat binary_little_endian 1.0\n' f'element vertex {len(pos)}\n'
             + ''.join(f'property float {k}\n' for k in cols) + 'end_header\n').encode())
    f.write(arr.tobytes())
print(f'{len(pos):,} Gaussians -> {dst}')
