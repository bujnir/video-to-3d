# Video to 3D: the room-modelling pipeline

Everything for turning a video (or photos) of a room into 3D models, with our results on a test room.

- **Live demo page:** https://claude.ai/artifact/3a3w7dhJnJPhaDADsRnjqD (the same page is in `viewer/`)
- **Test data:** 104 photos of the Mip-NeRF 360 "room" scene (handheld camera, not a drone)

```
video ──► 1 frames ──► 2 camera poses ──► 3 Gaussian splat ──► 4 smooth surface ──► 5 shapes (SketchUp style)
          (CPU)        (CPU, COLMAP)      (GPU, RunPod)        (CPU)                 (CPU)
```

## Quick start: the app (Docker)

Pick a video, paste a RunPod key, get everything back.

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and start it.
2. Double-click `start.bat` (Windows) or run `./start.sh` (macOS/Linux). The first start builds the image (about 5 minutes, 3 GB); later starts take seconds.
3. Your browser opens http://localhost:8080. Choose the video, paste your RunPod API key, press **Start**.
4. Watch the steps tick off (typically 15–30 minutes). When it's done: **Open the 3D viewer** or **Download all files (.zip)**.

The key is used for that run only: it is passed to the pipeline in memory and sent only to RunPod's API, never saved or logged. To avoid typing it every time, put `RUNPOD_API_KEY=...` in a `.env` file next to `docker-compose.yml` (git ignores it). The GPU pod is always shut down at the end, also when you press **Cancel** or stop the app. Results stay in `runs/` next to this file. Stop the app with `docker compose down`.

What you get (in `runs/<run>/output/` and in the zip):

| Folder | Files |
| --- | --- |
| `viewer/` | `index.html`: photo-real model, smooth surface and shapes-over-photo, all interactive |
| `splat/` | `room.splat`, `room.ply` (standard 3D Gaussian Splatting), `flythrough.mp4`, quality renders |
| `surface/` | `surface_topography.glb` (height colours), `surface.obj`, `surface_full.ply` |
| `shapes/` | `shapes.dae` (SketchUp), `shapes.glb`, `shapes.obj`, `shapes_plan.png`, `shapes.json` |
| `cameras/` | `colmap/` (COLMAP model), `points.ply` (sparse point cloud), `cameras.json` |
| top level | `report.json` (stats, timings, GPU cost), `README.txt`, `logs/` |

Without Docker, the same in one command: `pip install -r requirements.txt flask`, then
`RUNPOD_API_KEY=... python app/pipeline.py my_room.mp4 --out runs/my_room` (or `python app/server.py` for the web app).

## Folder layout

| Folder | Code | Output |
| --- | --- | --- |
| `1_frames/` | `video_to_frames.py` (video → sharp frames), `download_data.py` (test photos instead), `viz_features.py` (step pictures for the page) | frames go to `work/img` |
| `2_camera_poses/` | `sfm.py` (COLMAP: where each photo was taken + sparse 3D points), `export.py` (point cloud for the page) | `output/sparse_model/` (COLMAP model), `output/log.json` |
| `3_gaussian_splat/` | `pack_for_gpu.py` (photos + COLMAP model → upload package, viewer cameras), `run_pod.py` (rent GPU, train, download, delete pod), `pod_server.py`, `train_job.sh`, `ckpt_to_web.py`, `ckpt_to_splat.py`, `splat_to_ply.py`, `gs_normalize.py` | `output/` (held-out comparison renders, quality stats, fly-through video, settings) |
| `4_surface/` | `splat_to_mesh.py` (depth fusion → smooth mesh), `mesh_to_web.py`, `export_desktop.py` | `output/room_surface_topography.glb` (height colours), `room_surface.obj`, `room_surface_full.ply`, previews |
| `5_shapes/` | `blockout.py` (floor plane + walls + furniture boxes) | `output/room_blocks.dae` (SketchUp), `.glb`, `.obj`, `blocks.json`, plan image |
| `viewer/` | `index.html` (the demo page for the test room), `splat_viewer.js`, `screenshot_viewer.js` | all data files the page loads |
| `app/` | `pipeline.py` (all steps in one command), `server.py` (the web app), `viewer_template.html` (results page) | |
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

## Running the steps one by one

`app/pipeline.py` runs exactly these commands; use them to rerun or tune a single step.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# ffmpeg is needed for video (macOS: brew install ffmpeg; Ubuntu: apt install ffmpeg)

# 1 + 2: frames and camera poses (CPU)
python 1_frames/video_to_frames.py my_room.mp4 work 2      # 2 frames per second
python 2_camera_poses/sfm.py work                          # add "sequential" for long videos

# 3: Gaussian splat on a rented GPU: cp .env.example .env and put your RunPod key in .env
python 3_gaussian_splat/pack_for_gpu.py work work/gpu_in   # upload package + viewer cameras
set -a; . ./.env; set +a
python 3_gaussian_splat/run_pod.py work/gpu_in/data.tgz work/gpu_out 15000
mkdir -p work/gpu_out/res && tar -xzf work/gpu_out/results.tgz -C work/gpu_out/res
gunzip -c work/gpu_out/res/room_full.splat.gz > work/gpu_out/res/room.splat
python 3_gaussian_splat/splat_to_ply.py work/gpu_out/res/room.splat work/room.ply

# 4: smooth surface (CPU)
DS=2 RMAX=2 MINW=2 SMAX=0.08 EDGE=0.08 SMOOTH=40 \
  python 4_surface/splat_to_mesh.py work/gpu_out/res/room.splat work/gpu_in/splat_view.json work/mesh 0.011
mkdir -p work/web && python 4_surface/mesh_to_web.py work/mesh/mesh.ply work/web 220000 work/gpu_in/splat_view.json

# 5: shapes (CPU), then desktop copies of the surface in the same coordinates
python 5_shapes/blockout.py work/web/room_mesh.b64.txt work/gpu_in/splat_view.json work/shapes
python 4_surface/export_desktop.py work/web work/gpu_in/splat_view.json work/shapes/room_frame.json
```

The demo page for the test room is in `viewer/` (`cd viewer && python -m http.server 8000`, then open http://localhost:8000). `1_frames/viz_features.py` and `2_camera_poses/export.py` make the extra pictures and point cloud used only by that demo page.

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
