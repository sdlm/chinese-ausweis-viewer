#!/bin/bash
http -d -o predict.jpg -f POST http://127.0.0.1:8000/predict/ file@./chinese_ausweis_viewer/data/train/original_sample_blur2_256.jpg && display ./predict.jpg && rm -rf predict.jpg
