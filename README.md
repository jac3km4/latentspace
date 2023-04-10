# latentspace
This library aims to be a a complete Stable Diffusion implementation in idiomatic Rust built on top of [ONNX](https://github.com/microsoft/onnxruntime). It's still in an early work-in-progress stage, but some pipelines already work, [see examples](/examples).

<img src="https://user-images.githubusercontent.com/11986158/230987298-519e3229-90bf-4ee2-99d6-bb7cd92952ea.png" width="160px"/><img src="https://user-images.githubusercontent.com/11986158/230987224-cc2c7d39-08c1-4616-b641-619d86a4db3b.png" width="160px"/><img src="https://user-images.githubusercontent.com/11986158/230987601-429c4d45-f780-4845-899f-1deb0971935d.png" width="160px"/><img src="https://user-images.githubusercontent.com/11986158/230987544-44fc4d2d-c5b3-4618-806e-42e884ad6f29.png" width="160px"/>

## features
###### Legend: 🐥 complete, 🐣 partially implemented, 🥚 not available yet

- 🐥 txt2img
- 🐣 img2img and inpainting
- 🐥 CUDA and TensorRT accelerated inference
    - txt2img completes in ~2 seconds for a 512x512 image on modern Nvidia hardware
- 🐥 fp32 and fp16 models
    - fp16 models can be leveraged to save space and memory
- 🐣 support different schedulers
    - Euler, EulerAncestral and LMS for now
- 🐥 long prompt weighting
- 🥚 upscaling
- 🥚 LORAs and textual inversions
- 🐥 small footprint, no heavy dependencies
    - basic txt2img binary is <2MB on linux
    <br><img src="https://user-images.githubusercontent.com/11986158/230815471-08366950-2118-43cd-b402-da22cba1cb1b.png" width="280px"/>

## preparing checkpoints
Checkpoints have to be converted to the ONNX format before use. I've built a convenience Docker image that can convert `safetensors` files into optimized fp16 ONNX models, it can be invoked through a script in this repository:
```bash
./utils/convert/convert.sh -i /home/me/checkpoints/CounterfeitAnime.safetensors -o /home/me/checkpoints/CounterfeitAnime-onnx
```

## examples
- [Text-to-image](/examples/txt2img.rs)
- [Image-to-image](/examples/img2img.rs)
