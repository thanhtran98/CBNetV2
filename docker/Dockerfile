# ARG PYTORCH="1.7.0"
# ARG CUDA="11.0"
# ARG CUDNN="8"

# FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
FROM ninja0/mmdet:pytorch1.7.1-py37-cuda11.0-openmpi-mmcv1.3.3-apex-timm

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# # Avoid Public GPG key error
# # https://github.com/NVIDIA/nvidia-docker/issues/1631
# RUN rm /etc/apt/sources.list.d/cuda.list \
#     && rm /etc/apt/sources.list.d/nvidia-ml.list \
#     && apt-key del 7fa2af80 \
#     && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
#     && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# RUN git clone https://github.com/NVIDIA/apex.git && cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--fast_multihead_attn" . && cd .. && rm -rf apex


# Install MMCV
RUN pip install mmcv-full==1.3.16 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
RUN pip install numpy==1.20.1
# RUN git clone https://github.com/NVIDIA/apex.git
# RUN cd apex
# RUN pip install -v --no-cache-dir .
# RUN cd ..
# RUN git clone https://github.com/NVIDIA/apex
# RUN cd apex
# RUN pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# RUN cd ..
# RUN git clone https://github.com/NVIDIA/apex.git && cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--fast_multihead_attn" . && cd .. && rm -rf apex
# RUN pip install apex

# Install MMDetection
RUN conda clean --all
# RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
RUN git clone https://github.com/thanhtran98/CBNetV2.git /mmdetection
WORKDIR /mmdetection
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .
