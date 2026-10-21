"""Convert a gsplat simple_trainer checkpoint (.pt) to the web .splat format and a standard 3DGS .ply.

Usage: python ckpt_to_splat.py ckpt.pt out.splat out.ply [max_splats]
"""
import sys, json
import numpy as np
import torch

ck, dst, ply_out = sys.argv[1], sys.argv[2], sys.argv[3]
limit = int(sys.argv[4]) if len(sys.argv) > 4 else 0
s = torch.load(ck, map_location='cpu')['splats']
means = s['means'].numpy()
scales_log = s['scales'].numpy()
quats = s['quats'].numpy()
opac_logit = s['opacities'].numpy()
sh0 = s['sh0'].numpy().reshape(len(means), 3)
shN = s['shN'].numpy().reshape(len(means), -1, 3) if 'shN' in s else np.zeros((len(means), 0, 3))

# standard 3DGS .ply (for other viewers / tools)
n = len(means)
rest = shN.transpose(0, 2, 1).reshape(n, -1)  # channel-major like the reference implementation
cols = [means, np.zeros_like(means), sh0, rest, opac_logit[:, None], scales_log, quats]
names = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2'] + [f'f_rest_{i}' for i in range(rest.shape[1])] + ['opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
arr = np.concatenate(cols, 1).astype('<f4')
with open(ply_out, 'wb') as f:
    f.write(('ply\nformat binary_little_endian 1.0\nelement vertex %d\n' % n).encode())
    for nm in names:
        f.write(f'property float {nm}\n'.encode())
    f.write(b'end_header\n')
    f.write(arr.tobytes())

# compact .splat for the web viewer
scale = np.exp(scales_log)
alpha = 1 / (1 + np.exp(-opac_logit))
rgb = 0.5 + 0.28209479177387814 * sh0
q = quats / np.linalg.norm(quats, axis=1, keepdims=True)
order = np.argsort(-(scale.prod(1) * alpha))
if limit and n > limit:
    order = order[:limit]
out = np.zeros(len(order), dtype=[('p', '<f4', 3), ('s', '<f4', 3), ('c', 'u1', 4), ('r', 'u1', 4)])
out['p'] = means[order]
out['s'] = scale[order]
out['c'][:, :3] = np.clip(rgb[order] * 255, 0, 255).astype(np.uint8)
out['c'][:, 3] = np.clip(alpha[order] * 255, 0, 255).astype(np.uint8)
out['r'] = np.clip(q[order] * 128 + 128, 0, 255).astype(np.uint8)
out.tofile(dst)
print(json.dumps({'splats_total': int(n), 'kept': int(len(order)), 'splat_bytes': int(out.nbytes)}))
