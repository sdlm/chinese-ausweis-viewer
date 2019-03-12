# chinese-ausweis-viewer
Implement UNET for crop chinese card id. 
And tesseract for OCR card fields.

# dependencies
- Tensorflow 1.13

# run script
```bash
docker run --runtime=nvidia \
           -p 80:8888 \
           -v /root/chinese-ausweis-viewer/chinese_ausweis_viewer/:/tf/ \
           sdlmer/ausweis:notebook
```
