# Deep 360° Optical Flow Estimation Based on Multi-Projection Fusion

This project is the official repository of the ECCV2022 paper: Deep 360° Optical Flow Estimation Based on Multi-Projection Fusion. 

Authors: Yiheng Li, Connelly Barnes, Kun Huang, and Fang-Lue Zhang

[paper](https://arxiv.org/abs/2208.00776), [dataset](https://vuw-my.sharepoint.com/:f:/g/personal/zhangfa_staff_vuw_ac_nz/Eh6vF7Ge17hJmjiZ4L8nja8BWihQg1CVhkRWiDwHkmg0Ug?e=epeqCH)

### Abstract 
Optical flow computation is essential in the early stages of the video processing pipeline. This paper focuses on a less explored problem in this area, the 360° optical flow estimation using deep neural networks to support increasingly popular VR applications. To address the distortions of panoramic representations when applying convolutional neural networks, we propose a novel multi-projection fusion framework that fuses the optical flow predicted by the models trained using different projection methods. It learns to combine the complementary information in the optical flow results under different projections. We also build the first large-scale panoramic optical flow dataset to support the training of neural networks and the evaluation of panoramic optical flow estimation methods. The experimental results on our dataset demonstrate that our method outperforms the existing methods and other alternative deep networks that were developed for processing 360° content.

### Poster
<img align="left" src="poster_and_video/poster.png">

### Video
<video src="poster_and_video/video.mp4" width="640px" height="360px" controls="controls"></video>

### Citation
```
@misc{2208.00776,
Author = {Yiheng Li and Connelly Barnes and Kun Huang and Fang-Lue Zhang},
Title = {Deep 360$^\circ$ Optical Flow Estimation Based on Multi-Projection Fusion},
Year = {2022},
Eprint = {arXiv:2208.00776},
}
```

### Requirements
We mainly borrowed code from the PWC optical flow neural network and updated it to the newest Pytorch version. Furthermore, in order to accelerate the process of converting different projections. We also require C++ and OpenCL environment for parallel computing.

#### Python side:
Please refer to the `requirements.txt`. It requires:
Python=3.9
Pytorch=1.12
CUDA=11.6
OpenCV
NumPy
and the nvcc toolchain

#### C++/OpenCL side:
Please refer to the `CMakeLists.txt`. It requires:
cmake>=3.4
cc/c++
opencv=4.6
opencl=2.2
pybind11

### Installation
1. On Ubuntu 20.04, enter the project folder and execute the script `install.sh`. It will install the CUDA operator.

2. build the OpenCL code and port the dynamic file to the project folder
```bash
mkdir build && cd build && cmake .. && make -j8
```

### Inference
Please update the input arg in the `end_to_end_inference.py` for the model path and the dataset path.
```bash
python end_to_end_inference.py
```

### Contact 
Please feel free to contact me (Yiheng) at leehenry0314@me.com. Or raise your query in the GitHub issue :)

### Known issues
I have optimized the projection algorithm from pure C++ to OpenCL, and it caused some loss in precision. Further work will include adding an end-to-end training script and fine-tuning some fusion models.

