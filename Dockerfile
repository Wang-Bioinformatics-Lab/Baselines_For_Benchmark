FROM nvidia/cuda:11.6.2-base-ubuntu20.04

COPY requirements.txt /
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip install -r /requirements.txt
RUN pip install jupyterlab
RUN pip install jupyterlab-git
ADD mnist_test mnist_test