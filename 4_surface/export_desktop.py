"""Write desktop-friendly copies of the web mesh: Y-up, floor at height 0, height colours baked in.
Usage: python export_desktop.py web_dir splat_view.json [room_frame.json]
With room_frame.json (from blockout.py) the surface uses the same coordinates as the block model."""
import base64, json, struct, sys
import numpy as np, trimesh
d, view = sys.argv[1], json.load(open(sys.argv[2]))
blob = base64.b64decode(open(f'{d}/room_mesh.b64.txt').read())
nv, nf = struct.unpack('<II', blob[:8]); lo = np.array(struct.unpack('<3f', blob[8:20])); hi = np.array(struct.unpack('<3f', blob[20:32]))
V = lo + (hi - lo) * np.frombuffer(blob, '<u2', nv * 3, 32).reshape(-1, 3) / 65535
F = np.frombuffer(blob, '<u4', nf * 3, 32 + nv * 6).reshape(-1, 3)
if len(sys.argv) > 3:   # shared room frame: floor plane at y = 0, x/z along the walls
    fr = json.load(open(sys.argv[3])); ax, n, az = (np.array(fr[k]) for k in ('ax', 'n', 'az'))
    Vr = np.stack([V @ ax, V @ n + fr['d'], V @ az], 1)
    h = Vr[:, 1]; h0, h1 = 0.0, np.percentile(h, 99.5)
else:
    up = np.array(view['up']); up /= np.linalg.norm(up)
    x = np.cross(up, [1, 0, 0]); x /= np.linalg.norm(x); z = np.cross(x, up)
    Vr = np.stack([V @ x, V @ up, V @ z], 1)          # new Y = room up
    h = Vr[:, 1]; h0, h1 = np.percentile(h, 1), np.percentile(h, 99.5)
    Vr[:, 1] -= h0; Vr[:, [0, 2]] -= np.median(Vr[:, [0, 2]], 0)
t = np.clip((h - h0) / (h1 - h0), 0, 1) * 4
stops = np.array([[34, 87, 122], [56, 163, 165], [128, 237, 153], [242, 204, 143], [224, 122, 95]], float)
i = np.minimum(t.astype(int), 3); f = (t - i)[:, None]
col = np.concatenate([(stops[i] * (1 - f) + stops[i + 1] * f), np.full((nv, 1), 255)], 1).astype(np.uint8)
m = trimesh.Trimesh(Vr, F, vertex_colors=col, process=False)
m.export(f'{d}/room_surface_topography.glb')
trimesh.Trimesh(Vr, F, process=False).export(f'{d}/room_surface.obj')
print('ok', nv, nf)
