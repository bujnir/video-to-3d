set -euo pipefail
cd /workspace
python -c "import torch; assert torch.cuda.is_available(); print('cuda ok', torch.cuda.get_device_name(0))"
export TORCH_CUDA_ARCH_LIST="8.9"
tar -xzf data.tgz; ls data/sparse/0
pip install -q "gsplat==1.4.0" --index-url https://docs.gsplat.studio/whl/pt24cu124 --extra-index-url https://pypi.org/simple || pip install -q "gsplat==1.4.0"
git clone -q --depth 1 --branch v1.4.0 https://github.com/nerfstudio-project/gsplat.git gs
cd gs/examples
grep -v fused-ssim requirements.txt > req2.txt
pip install -q -r req2.txt
pip install -q --no-build-isolation "git+https://github.com/rahul-goel/fused-ssim@1272e21a282342e89537159e4bad508b19b34157"
echo "== train"
START=$(date +%s)
python simple_trainer.py default --data_dir /workspace/data --data_factor 1 --result_dir /workspace/out \
  --max_steps 15000 --eval_steps 15000 --save_steps 15000 --disable_viewer
echo "TRAIN_SECONDS $(( $(date +%s) - START ))" | tee /workspace/out/train_seconds.txt
CK=$(ls -t /workspace/out/ckpts/*.pt | head -1)
python /workspace/ckpt_to_web.py "$CK" /workspace/out/web 2 0.005
python /workspace/ckpt_to_splat.py "$CK" /workspace/out/room_full.splat /workspace/out/room.ply 0
cd /workspace/out && tar -czf /workspace/results.tgz --exclude='ckpts' --exclude='room.ply' --exclude='tb' --exclude='room_full.splat' .
gzip -k -1 /workspace/out/room_full.splat
ls -la /workspace/out /workspace/out/web; du -sh /workspace/results.tgz /workspace/out/room_full.splat.gz
echo ALL_DONE
