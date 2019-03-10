FROM tensorflow/tensorflow:1.13.1-py3-jupyter

RUN apt update && apt install -y libjpeg-dev libjpeg8-dev

RUN pip install --no-cache-dir keras imageio flask gunicorn

WORKDIR /src
