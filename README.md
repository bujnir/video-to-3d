# Video to 3D: the room-modelling pipeline

Everything for turning a video (or photos) of a room into 3D models, with our results on a test room.

- **Live demo page:** https://claude.ai/artifact/3a3w7dhJnJPhaDADsRnjqD (the same page is in `viewer/`)
- **Test data:** 104 photos of the Mip-NeRF 360 "room" scene (handheld camera, not a drone)

```
video ──► 1 frames ──► 2 camera poses ──► 3 Gaussian splat ──► 4 smooth surface ──► 5 shapes (SketchUp style)
          (CPU)        (CPU, COLMAP)      (GPU, RunPod)        (CPU)                 (CPU)
```

## Folder layout

| Folder | Code | Output |
| --- | --- | --- |
| `1_frames/` | `video_to_frames.py` (video → sharp frames), `download_data.py` (test photos instead), `viz_features.py` (step pictures for the page) | frames go to `work/img` |
| `2_camera_poses/` | `sfm.py` (COLMAP: where each photo was taken + sparse 3D points), `export.py` (point cloud for the page) | `output/sparse_model/` (COLMAP model), `output/log.json` |
| `3_gaussian_splat/` | `run_pod.py` (one command: rent GPU, train, download, delete pod), `pod_server.py`, `job3.sh`, `ckpt_to_web.py`, `ckpt_to_splat.py`, `gs_normalize.py` | `output/` (held-out comparison renders, quality stats, fly-through video, settings) |
| `4_surface/` | `splat_to_mesh.py` (depth fusion → smooth mesh), `mesh_to_web.py`, `export_desktop.py` | `output/room_surface_topography.glb` (height colours), `room_surface.obj`, `room_surface_full.ply`, previews |
| `5_shapes/` | `blockout.py` (floor plane + walls + furniture boxes) | `output/room_blocks.dae` (SketchUp), `.glb`, `.obj`, `blocks.json`, plan image |
| `viewer/` | `index.html` (the demo page), `splat_viewer.js`, `screenshot_viewer.js` | all data files the page loads |
| `archive/` | earlier README and a downloaded copy of an older page version (local only, not in git) | |

`work/` and `runs/` (created when you run the steps) hold frames and intermediate files; git ignores them, as well as the API key file `.env`.

## Results on the test room

| Step | Result | Time / cost |
| --- | --- | --- |
| 2 Camera poses | 104 of 104 photos solved, 17,524 points, 0.52 px average error | ~6 min on 2 CPU cores |
| 3 Gaussian splat | 1,023,804 Gaussians; PSNR 27.9 dB, SSIM 0.867 on 13 photos it never saw (published: ~31 dB with all photos and 2x training) | 7.5 min on an RTX 4090, ~$0.45 |
| 4 Smooth surface | 220,000 triangles, coloured by height | 3.8 min CPU |
| 5 Shapes | room shell + 63 boxes (sofa, armchair, tables with legs, cabinets) | seconds on CPU |

All models are in relative units: photos alone have no scale. Add one known distance, IMU data, or a depth/LiDAR sensor to get real measurements.

## How to run it on your own video

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# ffmpeg is needed for video (macOS: brew install ffmpeg; Ubuntu: apt install ffmpeg)

# 1 + 2: frames and camera poses (CPU)
python 1_frames/video_to_frames.py my_room.mp4 work 2      # 2 frames per second
python 2_camera_poses/sfm.py work
python 1_frames/viz_features.py work                       # optional step pictures
python 2_camera_poses/export.py work                       # point cloud for the page

# 3: Gaussian splat on a rented GPU: cp .env.example .env, put your RunPod key in .env (git ignores it)
#    data.tgz = data/images/*.jpg + data/sparse/0/{cameras,images,points3D}.bin
set -a; . ./.env; set +a
python 3_gaussian_splat/run_pod.py data.tgz runs/room1

# 4: smooth surface (CPU)
DS=2 RMAX=2 MINW=2 SMAX=0.08 EDGE=0.08 SMOOTH=40 \
  python 4_surface/splat_to_mesh.py runs/room1/room_full.splat viewer/splat_view.json runs/mesh 0.011
python 4_surface/mesh_to_web.py runs/mesh/mesh.ply viewer 220000 viewer/splat_view.json

# 5: shapes (CPU), then desktop copies of the surface in the same coordinates
python 5_shapes/blockout.py viewer/room_mesh.b64.txt viewer/splat_view.json 5_shapes/output
python 4_surface/export_desktop.py viewer viewer/splat_view.json 5_shapes/output/room_frame.json

# look at it
cd viewer && python -m http.server 8000                    # open http://localhost:8000
```

### Capture tips (from the test)
- Move slowly; motion blur is the main cause of failed matches.
- 70–80% overlap between neighbouring frames. With every 6th test photo only 29 of 52 joined; with every 3rd, all 104 did.
- Good light and visual detail. Plain white walls give almost no features.
- Don't only spin in place: the camera must also move (e.g. two heights).

## Opening the files

- **SketchUp:** File → Import → `5_shapes/output/room_blocks.dae`.
- **Blender:** File → Import → glTF (`.glb`) for both `room_blocks.glb` and `room_surface_topography.glb`; they share coordinates and line up.
- **Quick look on Windows:** double-click a `.glb` (3D Viewer), or drag it onto https://gltf-viewer.donmccurdy.com.
- **Splat:** the full `.splat` file (33 MB) was sent in the chat, as it's too big for the folder sync. Open it at https://superspl.at/editor.

## Notes and limits

- **RunPod:** use SECURE cloud (two community pods had broken CUDA drivers); `run_pod.py` always deletes the pod at the end.
- **Speed-up plan for step 3:** prebuilt Docker image (saves ~4 min of setup), GPU matching for video, 7k-step previews. Target ~5 min per room.
- **Surface:** holes remain where the photos saw little (behind furniture, window area). Open3D's TSDF returned empty meshes in our environment, so the fusion is plain NumPy.
- **Shapes:** furniture is fitted as axis-aligned boxes; anything taller than 80% of the wall height counts as wall. Objects with floor visible underneath become tables (top + 4 legs).

## Data credit and licenses

Test photos: Mip-NeRF 360 "room" (Barron et al.), from the `nvs-bench/mipnerf360` copy on Hugging Face.

All tools allow commercial use: pycolmap/COLMAP (BSD), gsplat (Apache 2.0), fused-ssim (MIT), Open3D (MIT), trimesh (MIT), scikit-image (BSD), three.js (MIT), Pillow (HPND), NumPy/SciPy (BSD), ffmpeg (LGPL/GPL, run as a separate program). See `../context/PROJECT_CONTEXT.md` section 6.
