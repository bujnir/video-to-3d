"""Mesh (.ply) -> compact web file (base64 text) for the demo page. Desktop files: export_desktop.py.

Usage: python mesh_to_web.py mesh.ply out_dir [max_triangles] [splat_view.json]
With splat_view.json, stray geometry outside the room footprint is cropped first.
Web layout (little endian), base64-encoded into room_mesh.b64.txt:
  uint32 n_vertices, uint32 n_triangles, float32 min[3], float32 max[3]
  uint16 positions[n_vertices*3]  (quantised inside min..max)
  uint32 indices[n_triangles*3]
"""
import base64, json, struct, sys
import numpy as np
import open3d as o3d

src, out = sys.argv[1], sys.argv[2]
max_tri = int(sys.argv[3]) if len(sys.argv) > 3 else 250_000
m = o3d.io.read_triangle_mesh(src)
if len(sys.argv) > 4:
    up = np.array(json.load(open(sys.argv[4]))['up']); up /= np.linalg.norm(up)
    V0 = np.asarray(m.vertices)
    a = np.cross(up, [1, 0, 0]); a /= np.linalg.norm(a); b = np.cross(up, a)
    H2 = np.stack([V0 @ a, V0 @ b], 1)
    c = np.median(H2, 0); ev, evec = np.linalg.eigh(np.cov((H2 - c).T))
    L = (H2 - c) @ evec  # room-aligned horizontal coordinates
    lo2, hi2 = np.percentile(L, 1.5, 0), np.percentile(L, 98.5, 0)
    pad = 0.04 * (hi2 - lo2)
    outside = np.any((L < lo2 - pad) | (L > hi2 + pad), 1)
    m.remove_vertices_by_mask(outside)
    cl, n_tri, _ = m.cluster_connected_triangles(); cl, n_tri = np.asarray(cl), np.asarray(n_tri)
    m.remove_triangles_by_mask(n_tri[cl] < 0.003 * len(cl)); m.remove_unreferenced_vertices()
    print(f'cropped {int(outside.sum()):,} stray vertices')
if len(m.triangles) > max_tri:
    m = m.simplify_quadric_decimation(max_tri)
m.remove_degenerate_triangles(); m.remove_unreferenced_vertices(); m.compute_vertex_normals()
V = np.asarray(m.vertices).astype(np.float32); F = np.asarray(m.triangles).astype(np.uint32)
lo, hi = V.min(0), V.max(0)
Q = np.clip(np.round((V - lo) / (hi - lo) * 65535), 0, 65535).astype('<u2')
blob = struct.pack('<II3f3f', len(V), len(F), *lo, *hi) + Q.tobytes() + F.astype('<u4').tobytes()
open(f'{out}/room_mesh.b64.txt', 'w').write(base64.b64encode(blob).decode())
info = {'vertices': int(len(V)), 'triangles': int(len(F)), 'web_bytes': len(blob)}
json.dump(info, open(f'{out}/room_mesh_web.json', 'w')); print(info)
