# Runs on the RunPod GPU pod (started by run_pod.py). STEPS is set by run_pod.py (default 15000).
set -euo pipefail
cd /workspace
STEPS="${STEPS:-15000}"
python -c "import torch; assert torch.cuda.is_available(); print('cuda ok', torch.cuda.get_device_name(0))"
# compile the CUDA parts only for the GPU this pod actually has (faster than all architectures)
export TORCH_CUDA_ARCH_LIST="$(python -c "import torch; print('%d.%d' % torch.cuda.get_device_capability())")"
echo "== stage: setup (arch $TORCH_CUDA_ARCH_LIST)"
tar -xzf data.tgz; ls data/sparse/0; echo "images: $(ls data/images | wc -l)"
pip install -q "gsplat==1.4.0" --index-url https://docs.gsplat.studio/whl/pt24cu124 --extra-index-url https://pypi.org/simple || pip install -q "gsplat==1.4.0"
git clone -q --depth 1 --branch v1.4.0 https://github.com/nerfstudio-project/gsplat.git gs
cd gs/examples
grep -v fused-ssim requirements.txt > req2.txt
pip install -q -r req2.txt
pip install -q --no-build-isolation "git+https://github.com/rahul-goel/fused-ssim@1272e21a282342e89537159e4bad508b19b34157"
echo "== stage: train ($STEPS steps)"
START=$(date +%s)
python simple_trainer.py default --data_dir /workspace/data --data_factor 1 --result_dir /workspace/out \
  --max_steps "$STEPS" --eval_steps "$STEPS" --save_steps "$STEPS" --disable_viewer
echo "TRAIN_SECONDS $(( $(date +%s) - START ))" | tee /workspace/out/train_seconds.txt
echo "== stage: convert"
CK=$(ls -t /workspace/out/ckpts/*.pt | head -1)
python /workspace/ckpt_to_web.py "$CK" /workspace/out/web 2 0.005
python /workspace/ckpt_to_splat.py "$CK" /workspace/out/room_full.splat /workspace/out/room.ply 0
gzip -1 -f /workspace/out/room_full.splat
cd /workspace/out && tar -czf /workspace/results.tgz --exclude='ckpts' --exclude='room.ply' --exclude='tb' .
du -sh /workspace/results.tgz
echo ALL_DONE
