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