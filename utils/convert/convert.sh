#!/bin/bash

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -i|--input) input="$2"; shift ;;
        -o|--output) output="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "${input}" ]; then
    echo "--input parameter is required"
    exit 1
fi

if [ ! -d "${output:=./output}" ]; then
  echo "Creating output directory ${output}"
  mkdir -p $output
fi

IS_SAFETENSORS=$([[ $input == *.safetensors ]] && echo "true" || echo "false")

COMMAND=$(hash podman 2>/dev/null && echo "podman" || echo "docker")

$COMMAND run --rm \
    --mount type=bind,src=${input},dst=/data/input,readonly \
    --mount type=bind,src=${output},dst=/data/output \
    --mount type=volume,src=hf-cache,dst="/data/cache" \
    -e IS_SAFETENSORS=${IS_SAFETENSORS} \
    -e HF_HOME=/data/cache \
    -e CONVERSION_INPUT=/data/input \
    -e CONVERSION_OUTPUT=/data/output \
    ghcr.io/jac3km4/sd-to-onnx:0.1.0
