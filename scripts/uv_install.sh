set -x




# symlinks
# ln -s /data/chenshuailin/checkpoints/llmc checkpoints
# mkdir -p /data/chenshuailin/
# if [ -d "/ms/FM/checkpoints" ]; then
#     ln -s /ms/FM/checkpoints /data/chenshuailin/checkpoints
#     ln -s /ms/FM/chenweixuan/r1-bf16 /data/chenshuailin/checkpoints/deepseek-ai/DeepSeek-R1-bf16
#     # ln -s /ms/AIED/lixiao/r1 /data/chenshuailin/checkpoints/deepseek-ai/DeepSeek-R1-fp8
# fi
# ln -s /ms/FM/chenshuailin/code/llmc ~/
# ln -s /ms/FM/chenshuailin/code/open_clip ~/

# dependencies
uv pip install -r requirements.txt

uv pip install flash-attn --no-build-isolation
uv pip install -U transformers datasets

uv pip install opencv-fixer==0.2.5
python -c "from opencv_fixer import AutoFix; AutoFix()"

uv pip uninstall qtorch
uv pip install -U qtorch

# fast-hadamard-transform
if [[ ! -d "fast-hadamard-transform" ]]; then
    git clone git@github.com:Dao-AILab/fast-hadamard-transform.git
fi
cd fast-hadamard-transform
uv pip install --no-build-isolation -e .


uv pip install psutil accelerate easydict debugpy loguru Pillow torchvision human_eval librosa natsort compressed-tensors
lmms_eval==0.3.4 

# # other fixes
# if [ -f "/usr/local/lib/python3.12/dist-packages/accelerate/utils/modeling.py" ]; then
#     cp site-packages.bk/modeling.py /usr/local/lib/python3.12/dist-packages/accelerate/utils/
# elif [ -f "/usr/local/lib/python3.10/dist-packages/accelerate/utils/modeling.py" ]; then
#     cp site-packages.bk/modeling.py /usr/local/lib/python3.10/dist-packages/accelerate/utils/
# else
#     echo "No site-packages found"
# fi
