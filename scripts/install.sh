set -x

# library path
if [ -d "/usr/local/cuda-12.2" ]; then
    export PATH=/usr/local/cuda-12.2/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
    export CUDA_PATH=/usr/local/cuda-12.2
fi

# symlinks
ln -s /data/chenshuailin/checkpoints/llmc checkpoints
mkdir -p /data/chenshuailin/
if [ -d "/ms/FM/checkpoints" ]; then
    ln -s /ms/FM/checkpoints /data/chenshuailin/checkpoints
    ln -s /ms/FM/chenweixuan/r1-bf16 /data/chenshuailin/checkpoints/deepseek-ai/DeepSeek-R1-bf16
    # ln -s /ms/AIED/lixiao/r1 /data/chenshuailin/checkpoints/deepseek-ai/DeepSeek-R1-fp8
fi
ln -s /ms/FM/chenshuailin/code/llmc ~/
ln -s /ms/FM/chenshuailin/code/open_clip ~/

# dependencies
pip install -r requirements.txt

pip install flash-attn --no-build-isolation
pip install -U transformers datasets

pip install opencv-fixer==0.2.5
python -c "from opencv_fixer import AutoFix; AutoFix()"

pip uninstall -y qtorch
pip install -U qtorch

cd ../fast-hadamard-transform
pip install --no-build-isolation -e .

# other fixes
if [ -f "/usr/local/lib/python3.12/dist-packages/accelerate/utils/modeling.py" ]; then
    cp site-packages.bk/modeling.py /usr/local/lib/python3.12/dist-packages/accelerate/utils/
elif [ -f "/usr/local/lib/python3.10/dist-packages/accelerate/utils/modeling.py" ]; then
    cp site-packages.bk/modeling.py /usr/local/lib/python3.10/dist-packages/accelerate/utils/
else
    echo "No site-packages found"
fi
