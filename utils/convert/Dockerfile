ARG PYTHON_VERSION=3.10
FROM docker.io/python:${PYTHON_VERSION}-slim

ARG ONNX_VERSION=1.14.1
ARG TORCH_VERSION=1.13.1
ARG DIFFUSERS_VERSION=0.13.0

RUN apt-get update && apt-get install git wget -y

RUN git clone --depth 1 --filter=blob:none --sparse https://github.com/huggingface/diffusers --branch v${DIFFUSERS_VERSION} \
    && cd diffusers \
    && git sparse-checkout set scripts \
    && cd ..

RUN git clone --depth 1 --filter=blob:none --sparse https://github.com/microsoft/onnxruntime \
    && cd onnxruntime \
    && git sparse-checkout set onnxruntime/python \
    && cd ..

RUN pip install --no-cache-dir \
    -r onnxruntime/onnxruntime/python/tools/transformers/models/stable_diffusion/requirements.txt \
    onnxruntime==${ONNX_VERSION} \
    torch==${TORCH_VERSION} \
    diffusers==${DIFFUSERS_VERSION} \
    safetensors \
    omegaconf

COPY entrypoint.sh .

ENV ONNX_OPSET=17

ENTRYPOINT ["/bin/bash", "entrypoint.sh"]
