import io

import pandas as pd
import numpy as np
from PIL import Image

from .helpers import resize_to_256
from .cutter import crop_by_mask
from .optical_character_recognition import extract_data, get_actual_box_coords
from .unet import get_compiled_model


h_size = 256
w_size = 256
start_neurons = 16


def load_boxes():
    df = pd.read_csv('./src/data/resident_identity_card_mapping.csv')
    boxes_ = list()
    for i in range(df.shape[0]):
        x0 = df.iloc[i][0]
        y0 = df.iloc[i][1]
        sx = df.iloc[i][2]
        sy = df.iloc[i][3]

        box = np.array([
            [x0, y0 + sy],
            [x0, y0],
            [x0 + sx, y0],
            [x0 + sx, y0 + sy],
        ])
        boxes_.append(box)
    return boxes_


boxes = load_boxes()


def get_model():
    model = get_compiled_model(h_size, w_size, start_neurons)
    model.load_weights('./src/data/weights/model_v1.h5')
    return model


def predict_mask(image_data):
    model = get_model()

    # prepare input image
    test_x = np.zeros((1, h_size, w_size, 3), dtype='float32')
    test_image = Image.open(io.BytesIO(image_data))
    test_x[0] = np.array(test_image, dtype='float32') / 255.0

    # predict mask
    test_y = model.predict(test_x)

    # prepare output array
    test_mask = test_y[0] * 255
    test_mask = test_mask.reshape(256, 256)
    test_mask = test_mask.astype(np.uint8)

    return test_mask


def prepare_output_img(mask_arr):
    zero_layer = np.zeros((256, 256), dtype=np.uint8)
    stacked_img = np.stack((zero_layer, mask_arr, zero_layer), axis=-1)

    # make jpg image
    image = Image.fromarray(stacked_img, 'RGB')
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format='jpeg')
    imgByteArr = imgByteArr.getvalue()

    return imgByteArr


def extract_card_fields_data(input_file):
    # load original images
    input_image = Image.open(input_file)
    card_arr = np.array(input_image, dtype=np.uint8)

    card_arr_256 = resize_to_256(card_arr)
    mask_256 = predict_mask(card_arr_256)

    # get card image by mask_256
    cropped_card = crop_by_mask(mask_256, card_arr)

    # get box coord for card fields
    box_coords = get_actual_box_coords(boxes, cropped_card)

    # extract text data from card image
    data = extract_data(cropped_card, box_coords)

    return data
