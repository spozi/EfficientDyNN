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
3. Profile of each pretrained model