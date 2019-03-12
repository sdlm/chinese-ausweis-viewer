# chinese-ausweis-viewer
Implement UNET for crop chinese card id. 
And tesseract for OCR card fields.

install TF 1.13

nvidia-docker run -p 80:8888 -v /root/work/chinese-ausweis-viewer/chinese_ausweis_viewer/:/tf/ sdlmer/ausweis:notebook

docker run --runtime=nvidia -p 80:8888 -v /root/work/chinese-ausweis-viewer/chinese_ausweis_viewer/:/tf/ sdlmer/ausweis:notebook

docker run --runtime=nvidia -p 80:8888 tensorflow/tensorflow:1.13.1-gpu-py3-jupyter

1.13.1-gpu-py3-jupyter


