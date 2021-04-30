FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

RUN apt update && apt -y install sudo
RUN sudo apt -y install vim build-essential git libgtk2.0-dev libgl1-mesa-dev

COPY . /workspace

# Using SparseConvNet with cuda, requires cuda present in build time.
# So either set your docker runtime to nvidia and uncomment this, or
#  run run this command inside the container once running
#WORKDIR /workspace/SparseConvNet
#RUN bash develop.sh

WORKDIR /workspace
RUN pip install -e .

RUN mkdir -p /workspace/persistent_storage/checkpoints/
RUN mkdir -p /workspace/persistent_storage/runs/
