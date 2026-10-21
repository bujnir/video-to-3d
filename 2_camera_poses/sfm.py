"""Camera poses + sparse 3D points with COLMAP (pycolmap).

Usage: python sfm.py [work_folder] [matcher]
  work_folder  holds img/ (default ../work); results go to work_folder/sfm
  matcher      exhaustive (every pair, best for photo sets) or sequential (neighbouring
               frames plus a few far ones, much faster for video). Default: sequential
               for more than 150 images, otherwise exhaustive.
"""
import pycolmap, time, os, shutil, json, sys
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / 'work'
img = root / 'img'
out = root / 'sfm'
if out.exists():
    shutil.rmtree(out)
out.mkdir()
db = out / 'database.db'
log = {}

t = time.time()
ext = pycolmap.FeatureExtractionOptions()
ext.sift.max_num_features = 4096
rd = pycolmap.ImageReaderOptions()
rd.camera_model = 'SIMPLE_RADIAL'
pycolmap.extract_features(db, img, camera_mode=pycolmap.CameraMode.SINGLE,
                          reader_options=rd, extraction_options=ext)
log['extract_s'] = round(time.time() - t, 1)
print('extract', log['extract_s'], flush=True)

n_img = len(list(img.glob('*.jpg')) + list(img.glob('*.png')))
matcher = sys.argv[2] if len(sys.argv) > 2 else ('sequential' if n_img > 150 else 'exhaustive')
log['images'] = n_img; log['matcher'] = matcher
t = time.time()
if matcher == 'sequential':
    po = pycolmap.SequentialPairingOptions(); po.overlap = 20; po.quadratic_overlap = True
    pycolmap.match_sequential(db, pairing_options=po)
else:
    pycolmap.match_exhaustive(db)
log['match_s'] = round(time.time() - t, 1)
print('match', log['match_s'], flush=True)

t = time.time()
recs = pycolmap.incremental_mapping(db, img, out)
log['map_s'] = round(time.time() - t, 1)
print('map', log['map_s'], flush=True)

best = max(recs.values(), key=lambda r: r.num_reg_images())
best.write(out)
for k, r in recs.items():
    if r is not best: print(f'note: extra partial model {k} with {r.num_reg_images()} images ignored')
log['summary'] = best.summary()
log['num_models'] = len(recs)
log['reg_images'] = best.num_reg_images()
log['points3D'] = best.num_points3D()
print(best.summary())
json.dump(log, open(out / 'log.json', 'w'), indent=1)
