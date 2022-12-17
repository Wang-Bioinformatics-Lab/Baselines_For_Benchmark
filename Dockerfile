#from jupyter/datascience-notebook:lab-3.4.4
FROM gcr.io/tensorflow/tensorflow:latest-gpu-jupyter

COPY requirements.txt /
RUN pip install -r /requirements.txt
RUN pip install jupyterlab-git