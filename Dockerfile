FROM mcr.microsoft.com/azureml/onnxruntime:latest-cuda

ENV DEBIAN_FRONTEND=noninteractive TZ=Asia/Shanghai

RUN apt update || true
RUN apt install -y \
        build-essential \ 
        python3-pip \
	python3-opencv
RUN pip3 install \
        grpcio \
        protobuf \
        torch>=1.7 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install yolox -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . /workspace/

ENTRYPOINT cd /workspace && python3 light_service.py
