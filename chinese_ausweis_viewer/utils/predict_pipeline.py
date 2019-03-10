import io

import PIL
import numpy as np
from PIL import Image

from .unet import get_compiled_model


h_size = 256
w_size = 256
start_neurons = 16


def get_model():
    model = get_compiled_model(h_size, w_size, start_neurons)
    model.load_weights('./src/data/weights/model_v1.h5')
    return model


def predict_mask(image_data):
    model = get_model()

    # prepare input image
    test_x = np.zeros((1, h_size, w_size, 3), dtype='float32')
    test_image = Image.open(io.BytesIO(image_data))
    test_image_arr = np.array(test_image)
    test_x[0] = test_image_arr

    # predict mask
    test_y = model.predict(test_x)

    # prepare output array
    test_mask = test_y[0]
    test_mask = test_mask.reshape(256, 256)
    zero_layer = np.zeros((256, 256), dtype='float32')
    stacked_img = np.stack((zero_layer, test_mask, zero_layer), axis=-1)

    # make jpg image
    image = Image.fromarray(stacked_img, 'RGB')
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format='jpeg')
    imgByteArr = imgByteArr.getvalue()

    image.save('/src/data/predict.png', format='png')
    return imgByteArr
