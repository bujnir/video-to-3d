"""Package the photos and camera poses for GPU training, and write the viewer's camera file.

Usage: python pack_for_gpu.py work_folder out_dir
Reads  work/img/*.jpg and the COLMAP model in work/sfm (from 2_camera_poses/sfm.py).
Writes out_dir/data.tgz         data/images + data/sparse/0/{cameras,images,points3D}.bin
       out_dir/splat_view.json  photo camera poses in the splat's (normalised) world space
       out_dir/points.ply       the sparse 3D points as a point cloud

The .bin files are written in the classic COLMAP layout with our own writer, because the
COLMAP reader inside gsplat's examples is older than current pycolmap output.
"""
import io, json, struct, sys, tarfile
from pathlib import Path
import numpy as np
import pycolmap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gs_normalize import normalize  # same normalisation the gsplat trainer applies

work, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
rec = pycolmap.Reconstruction(str(work / 'sfm'))
INVALID = 2 ** 64 - 1


def cameras_bin():
    b = io.BytesIO(); b.write(struct.pack('<Q', len(rec.cameras)))
    for cid, c in sorted(rec.cameras.items()):
        p = np.asarray(c.params, np.float64)
        b.write(struct.pack('<iiQQ', cid, int(c.model.value), c.width, c.height)); b.write(p.tobytes())
    return b.getvalue()


def images_bin():
    b = io.BytesIO(); b.write(struct.pack('<Q', len(rec.images)))
    for iid, im in sorted(rec.images.items()):
        T = im.cam_from_world()
        q = np.asarray(T.rotation.quat, np.float64)        # pycolmap: x, y, z, w
        t = np.asarray(T.translation, np.float64)
        b.write(struct.pack('<i4d3di', iid, q[3], q[0], q[1], q[2], *t, im.camera_id))
        b.write(im.name.encode() + b'\x00')
        pts = im.points2D; b.write(struct.pack('<Q', len(pts)))
        for p in pts:
            pid = p.point3D_id if p.has_point3D() else INVALID
            b.write(struct.pack('<ddq', float(p.xy[0]), float(p.xy[1]), -1 if pid == INVALID else int(pid)))
    return b.getvalue()


def points_bin():
    b = io.BytesIO(); b.write(struct.pack('<Q', len(rec.points3D)))
    for pid, p in sorted(rec.points3D.items()):
        el = p.track.elements
        b.write(struct.pack('<Q3d3Bd', pid, *map(float, p.xyz), *map(int, p.color), float(p.error)))
        b.write(struct.pack('<Q', len(el)))
        for e in el: b.write(struct.pack('<ii', e.image_id, e.point2D_idx))
    return b.getvalue()


# ---- data.tgz for the pod
with tarfile.open(out / 'data.tgz', 'w:gz') as tar:
    for name, blob in (('cameras.bin', cameras_bin()), ('images.bin', images_bin()), ('points3D.bin', points_bin())):
        ti = tarfile.TarInfo(f'data/sparse/0/{name}'); ti.size = len(blob); tar.addfile(ti, io.BytesIO(blob))
    for im in sorted(rec.images.values(), key=lambda i: i.name):
        tar.add(work / 'img' / im.name, arcname=f'data/images/{im.name}')

# ---- viewer camera file, in the same normalised space gsplat trains in
ims = sorted(rec.images.values(), key=lambda i: i.name)
w2c = []
for im in ims:
    M = np.eye(4); M[:3, :] = im.cam_from_world().matrix(); w2c.append(M)
c2w = np.linalg.inv(np.stack(w2c))
pts = np.array([p.xyz for p in rec.points3D.values()], np.float64)
c2w_n, pts_n, T = normalize(c2w, pts)
cams = []
for im, C in zip(ims, c2w_n):
    R = C[:3, :3].T; t = -R @ C[:3, 3]
    cams.append({'name': im.name, 'R': np.round(R.reshape(-1), 6).tolist(), 't': np.round(t, 6).tolist()})
cam0 = rec.cameras[ims[0].camera_id]
ups = -c2w_n[:, :3, 1]; up = ups.mean(0); up /= np.linalg.norm(up)
centres = c2w_n[:, :3, 3]
view = {'fx': float(cam0.params[0]), 'w': cam0.width, 'h': cam0.height, 'up': np.round(up, 6).tolist(),
        'center': np.round(np.median(pts_n, 0), 4).tolist(), 'cam_center': np.round(centres.mean(0), 4).tolist(),
        'radius': round(float(np.linalg.norm(centres - centres.mean(0), axis=1).mean()) * 1.3, 4),
        'cams': cams, 'note': 'camera poses in gsplat simple_trainer normalized world space'}
json.dump(view, open(out / 'splat_view.json', 'w'))
json.dump({'matrix': np.round(T, 8).tolist(), 'scale': float(np.cbrt(abs(np.linalg.det(T[:3, :3]))))}, open(out / 'world_transform.json', 'w'))

# ---- sparse point cloud as .ply (original COLMAP coordinates)
cols = np.array([p.color for p in rec.points3D.values()], np.uint8)
with open(out / 'points.ply', 'wb') as f:
    f.write(f'ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\n'
            'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'.encode())
    rec_arr = np.zeros(len(pts), dtype=[('p', '<f4', 3), ('c', 'u1', 3)]); rec_arr['p'] = pts; rec_arr['c'] = cols
    f.write(rec_arr.tobytes())
print(json.dumps({'images': len(ims), 'points': len(pts), 'data_tgz_mb': round((out / 'data.tgz').stat().st_size / 1e6, 1)}))
