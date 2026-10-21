import pycolmap, time, os, shutil, json, sys
from pathlib import Path

# Work folder: first command-line argument, default ./work next to this script
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

t = time.time()
pycolmap.match_exhaustive(db)
log['match_s'] = round(time.time() - t, 1)
print('match', log['match_s'], flush=True)

t = time.time()
recs = pycolmap.incremental_mapping(db, img, out)
log['map_s'] = round(time.time() - t, 1)
print('map', log['map_s'], flush=True)

best = max(recs.values(), key=lambda r: r.num_reg_images())
best.write(out)
log['summary'] = best.summary()
log['num_models'] = len(recs)
log['reg_images'] = best.num_reg_images()
log['points3D'] = best.num_points3D()
print(best.summary())
json.dump(log, open(out / 'log.json', 'w'), indent=1)
