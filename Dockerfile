FROM tensorflow/tensorflow:1.13.1-py3-jupyter

RUN pip install --no-cache-dir keras imageio flask gunicorn

RUN mkdir -p /src/app/data/iploads

WORKDIR /src
