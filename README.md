# EfficientDyNN
EfficientDyNN is a Python library for efficient and lazy evaluation of dynamic neural networks with a focus on minimizing latency.

# Overview
Dynamic neural networks (dynn) often require efficient handling of latency, especially in real-time applications. EfficientDyNN provides a set of tools and techniques to optimize latency while maintaining the flexibility of dynamic neural networks. It leverages latent representations and lazy evaluation strategies to achieve efficient computation and resource utilization.

# Features
Efficient handling of latency in dynamic neural networks.
Utilization of latent representations for compact and informative feature extraction.
Lazy evaluation techniques for deferred computation and resource optimization.
Compatible with popular deep learning frameworks such as \st{TensorFlow} and PyTorch.

# Functions
1. Database of pretrained models (From [Onnx](https://github.com/onnx)):
    - SqueezeNet1.1
    - MobileNetV3
    - ResNet50
2. Build shared library for each model through cross compilation
    - Cross compilation is much easier if we build through docker, that is (I think):
        - Boot a linux based non x86-64 platform
        - Deploy a docker container
            - Install miniforge
            - Install tvm
        - Install necessary library
        - Save it as image and deploy it on x86-64 server
3. Profile of each pretrained model

# Requirements
1. TVM is build and installed in your environment.