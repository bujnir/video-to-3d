"""Convert a gsplat checkpoint into compact web parts (all splats, 17 bytes each).

Usage: python ckpt_to_web.py ckpt.pt out_dir [parts] [min_alpha]
Writes out_dir/room_partK.bin and out_dir/web_convert.json.

Part layout (little endian):
  header 48 bytes: uint32 count, uint32 version=1, float32 min[3], float32 max[3],
                   float32 log_scale_min, float32 log_scale_max, 8 bytes padding
  positions uint16[count*3]  (quantised inside min..max)
  log-scales uint8[count*3]  (quantised inside log_scale_min..max)
  rotations uint8[count*4]   (w,x,y,z, q*128+128)
  colours   uint8[count*4]   (r,g,b,alpha)
"""
import sys, json, struct, os
import numpy as np
import torch

ck, out_dir = sys.argv[1], sys.argv[2]
parts = int(sys.argv[3]) if len(sys.argv) > 3 else 2
min_alpha = float(sys.argv[4]) if len(sys.argv) > 4 else 0.005
s = torch.load(ck, map_location='cpu', weights_only=False)['splats']
means = s['means'].numpy().astype(np.float32)
ls = s['scales'].numpy().astype(np.float32)
q = s['quats'].numpy().astype(np.float32)
q /= np.linalg.norm(q, axis=1, keepdims=True)
alpha = 1 / (1 + np.exp(-s['opacities'].numpy()))
rgb = 0.5 + 0.28209479177387814 * s['sh0'].numpy().reshape(len(means), 3)

keep = alpha >= min_alpha
n_all = len(means)
means, ls, q, alpha, rgb = means[keep], ls[keep], q[keep], alpha[keep], rgb[keep]
n = len(means)
# drop extreme outliers in position (far floaters) so quantisation stays fine
lo, hi = np.percentile(means, 0.05, 0), np.percentile(means, 99.95, 0)
inside = np.all((means >= lo) & (means <= hi), 1)
means, ls, q, alpha, rgb = means[inside], ls[inside], q[inside], alpha[inside], rgb[inside]
n_kept = len(means)
lsmin, lsmax = float(np.percentile(ls, 0.01)), float(np.percentile(ls, 99.99))

P = np.clip(np.round((means - lo) / (hi - lo) * 65535), 0, 65535).astype('<u2')
S = np.clip(np.round((ls - lsmin) / (lsmax - lsmin) * 255), 0, 255).astype(np.uint8)
R = np.clip(np.round(q * 128 + 128), 0, 255).astype(np.uint8)
C = np.zeros((n_kept, 4), np.uint8)
C[:, :3] = np.clip(rgb * 255, 0, 255).astype(np.uint8)
C[:, 3] = np.clip(alpha * 255, 0, 255).astype(np.uint8)

os.makedirs(out_dir, exist_ok=True)
idx = np.array_split(np.random.default_rng(0).permutation(n_kept), parts)  # random split: each part covers the whole room
sizes = []
for k, ii in enumerate(idx):
    ii = np.sort(ii)
    hdr = struct.pack('<II3f3f2f8x', len(ii), 1, *lo, *hi, lsmin, lsmax)
    path = f'{out_dir}/room_part{k}.bin'
    with open(path, 'wb') as f:
        f.write(hdr); f.write(P[ii].tobytes()); f.write(S[ii].tobytes()); f.write(R[ii].tobytes()); f.write(C[ii].tobytes())
    sizes.append(os.path.getsize(path))
info = {'splats_total': int(n_all), 'splats_shown': int(n_kept), 'dropped_transparent': int(n_all - n), 'part_bytes': sizes}
json.dump(info, open(f'{out_dir}/web_convert.json', 'w'))
print(json.dumps(info))
