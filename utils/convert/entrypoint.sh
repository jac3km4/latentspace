#!/bin/bash

DIFFUSERS_TMP=./diffusers_output
ONNX_TMP=./onnx_output

if [ "$IS_SAFETENSORS" == "true" ]; then
    python ./diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py \
        --checkpoint_path=${CONVERSION_INPUT} \
        --dump_path=${DIFFUSERS_TMP} \
        --from_safetensors \
        --extract_ema
else
    cp ${CONVERSION_INPUT} ${DIFFUSERS_TMP}
fi

python ./diffusers/scripts/convert_stable_diffusion_checkpoint_to_onnx.py \
    --model_path=${DIFFUSERS_TMP} \
    --output_path=${ONNX_TMP} \
    --opset=${ONNX_OPSET}

rm -rf ${DIFFUSERS_TMP}

python ./onnxruntime/onnxruntime/python/tools/transformers/models/stable_diffusion/optimize_pipeline.py \
    -i ${ONNX_TMP} \
    -o ${CONVERSION_OUTPUT} \
    --float16
