#!/bin/bash
http -d -o predict.jpg -f POST http://127.0.0.1:8000/predict/ file@./chinese_ausweis_viewer/data/test/test_001.jpg && display ./predict.jpg && rm -rf predict.jpg
