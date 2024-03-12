# Start with the official Ubuntu base image (I am using Mac M1)
FROM --platform=linux/arm64 ubuntu:latest

# Set environment variables (if needed)
ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install necessary packages
RUN apt-get update && \
    apt-get install -y \
    software-properties-common \
    apt-utils \
    curl \
    wget \
    git \
    vim \
    openssh-server \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install ARM Compute Library
RUN apt-get update && \
    apt-get install -y \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    ocl-icd-opencl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -O ~/miniforge.sh && \
    /bin/bash ~/miniforge.sh -b -p /opt/conda && \
    rm ~/miniforge.sh

# Create Conda environment named 'dynn' with Python 3.8
RUN /opt/conda/bin/conda create -n dynn python=3.8 \
    cmake numpy decorator attrs typing-extensions psutil \
    scipy tornado cloudpickle xgboost pybind11 cython pythran ninja -y && \
    /opt/conda/bin/conda clean -afy

# Activate Conda environment
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Generate SSH keys
RUN mkdir /var/run/sshd && \
    echo 'root:root' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# Expose SSH port
EXPOSE 22

# Port forwarding from localhost:31000 to container:22
# This must be specified when running the container:
# docker run -d -p 31000:22 --rm -it --security-opt seccomp=unconfined <image_name>

# Default command to run when the container starts
CMD ["/usr/sbin/sshd", "-D"]