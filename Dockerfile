FROM mcr.microsoft.com/azureml/onnxruntime:latest-cuda

ENV DEBIAN_FRONTEND=noninteractive TZ=Asia/Shanghai

RUN apt update || true
RUN apt install -y \
        build-essential \ 
	python3-dev \
        python3-pip \
	python3-opencv
RUN python3 -m pip install pip --upgrade && \
    pip install \
        grpcio \
        protobuf \
        numpy \
        torch>=1.7 \
        opencv_python \
        loguru \
        scikit-image \
        tqdm \
        torchvision \
        Pillow \
        thop \
        ninja \
        tabulate \
        tensorboard \
        pycocotools>=2.0.2

COPY . /workspace/

ENTRYPOINT cd /workspace && python3 light_service.py
