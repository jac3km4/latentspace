# latentspace
This is library aims to be a a complete Stable Diffusion implementation in Rust on top of [ONNX](https://github.com/microsoft/onnxruntime) with support for state-of-the-art techniques like Textual Inversions and LORAs. It's still in an early work-in-progress stage, but some pipelines already work, [see examples](/examples).

<img src="https://user-images.githubusercontent.com/11986158/229587070-4e36f86f-426a-4fd2-91d5-bc936f839765.png" width="160px"/><img src="https://user-images.githubusercontent.com/11986158/229587077-50ac0148-921f-488f-98f2-a15786c7475b.png" width="160px"/>

## preparing checkpoints
Checkpoints have to be converted to the ONNX format before use. I've built a convenience Docker image that can convert `safetensors` files into optimized fp16 ONNX models, it can be invoked through a script in this repository:
```bash
./utils/convert/convert.sh -i /home/me/checkpoints/CounterfeitAnime.safetensors -o /home/me/checkpoints/CounterfeitAnime-onnx
```

## examples
See [txt2img.rs](/examples/txt2img.rs) for a text-to-image example.
