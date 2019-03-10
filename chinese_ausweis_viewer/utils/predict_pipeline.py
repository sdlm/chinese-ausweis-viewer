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
    test_x[0] = np.array(test_image, dtype='float32') / 255.0

    # predict mask
    test_y = model.predict(test_x)

    # prepare output array
    test_mask = test_y[0] * 255
    test_mask = test_mask.reshape(256, 256)
    test_mask = test_mask.astype(np.uint8)
    zero_layer = np.zeros((256, 256), dtype=np.uint8)
    stacked_img = np.stack((zero_layer, test_mask, zero_layer), axis=-1)

    # make jpg image
    image = Image.fromarray(stacked_img, 'RGB')
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format='jpeg')
    imgByteArr = imgByteArr.getvalue()

    return imgByteArr
