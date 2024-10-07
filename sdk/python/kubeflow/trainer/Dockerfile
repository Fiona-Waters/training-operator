## Global Args ######################################################
ARG BASE_UBI_IMAGE_TAG=latest
ARG PYTHON_VERSION=3.11

# use UBI9 latest
FROM registry.access.redhat.com/ubi9/ubi:${BASE_UBI_IMAGE_TAG} AS base

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt /app/requirements.txt

# Note this works for 3.9, 3.11, 3.12
RUN dnf remove -y --disableplugin=subscription-manager \
        subscription-manager \
    && dnf install -y python${PYTHON_VERSION} procps g++ python${PYTHON_VERSION}-devel \
    && python -m ensurepip --upgrade \
    && python -m pip install --upgrade pip \
    && python -m pip install --upgrade setuptools \
    && dnf update -y \
    && dnf clean all

## CUDA Base ###################################################################
FROM base AS cuda-base

# Ref: https://docs.nvidia.com/cuda/archive/12.1.0/cuda-toolkit-release-notes/
ENV CUDA_VERSION=12.1.0 \
    NV_CUDA_LIB_VERSION=12.1.0-1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NV_CUDA_CUDART_VERSION=12.1.55-1 \
    NV_CUDA_COMPAT_VERSION=530.30.02-1

RUN dnf config-manager \
    --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo \
 && dnf install -y \
     cuda-cudart-12-1-${NV_CUDA_CUDART_VERSION} \
     cuda-compat-12-1-${NV_CUDA_COMPAT_VERSION} \
 && echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
 && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf \
 && dnf clean all

ENV CUDA_HOME="/usr/local/cuda" \
 PATH="/usr/local/nvidia/bin:${CUDA_HOME}/bin:${PATH}" \
 LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"                         

## CUDA Development ############################################################
FROM cuda-base AS cuda-devel

# Ref: https://developer.nvidia.com/nccl/nccl-legacy-downloads
ENV NV_CUDA_CUDART_DEV_VERSION=12.1.55-1 \
    NV_NVML_DEV_VERSION=12.1.55-1 \
    NV_LIBCUBLAS_DEV_VERSION=12.1.0.26-1 \
    NV_LIBNPP_DEV_VERSION=12.0.2.50-1 \
    NV_LIBNCCL_DEV_PACKAGE_VERSION=2.18.3-1+cuda12.1

RUN dnf config-manager \
       --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo \
    && dnf install -y \
        cuda-command-line-tools-12-1-${NV_CUDA_LIB_VERSION} \
        cuda-libraries-devel-12-1-${NV_CUDA_LIB_VERSION} \
        cuda-minimal-build-12-1-${NV_CUDA_LIB_VERSION} \
        cuda-cudart-devel-12-1-${NV_CUDA_CUDART_DEV_VERSION} \
        cuda-nvml-devel-12-1-${NV_NVML_DEV_VERSION} \
        libcublas-devel-12-1-${NV_LIBCUBLAS_DEV_VERSION} \
        libnpp-devel-12-1-${NV_LIBNPP_DEV_VERSION} \
        libnccl-devel-${NV_LIBNCCL_DEV_PACKAGE_VERSION} \
    && dnf clean all

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python package and its source code into the container
COPY . /app

# Run hf_llm_training.py when the container launches
ENTRYPOINT ["torchrun", "hf_llm_training.py"]