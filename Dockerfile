FROM tensorflow/tensorflow:1.13.1-py3

RUN apt update && apt install -y software-properties-common && \
    add-apt-repository -y ppa:alex-p/tesseract-ocr && apt update

RUN apt install -y libsm6 libxext6 libxrender1 libglib2.0-0 libjpeg-dev libjpeg8-dev \
    python3-pil tesseract-ocr tesseract-ocr-eng tesseract-ocr-chi-sim

RUN pip install --no-cache-dir keras imageio tqdm imgaug \
                               opencv-python py-cpuinfo seaborn \
                               pytesseract opencv-python pillow==5.4.1 \
                               numpy pandas \
                               flask gunicorn requests lxml

WORKDIR /src
