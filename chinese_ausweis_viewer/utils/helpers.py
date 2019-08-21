from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


def show_predict_mask_above_image(test_image, model, h_size, w_size):
    test_x = np.zeros((1, h_size, w_size, 3), dtype='float32')
    test_x[0] = test_image

    # predict mask
    test_y = model.predict(test_x)

    # show predict mask above test image
    extent = 0, w_size, 0, h_size
    fig = plt.figure(frameon=False, figsize=(10, 10))
    im1 = plt.imshow(test_image, extent=extent)
    plt.axis('off')
    test_mask = test_y[0]
    test_mask = test_mask.reshape(256, 256)
    zero_layer = np.zeros((256, 256), dtype='float32')
    stacked_img = np.stack((zero_layer, test_mask, zero_layer), axis=-1)
    im2 = plt.imshow(stacked_img, alpha=0.5, extent=extent)


def resize_to_256(nd_array: np.ndarray) -> np.ndarray:
    return resize_to(nd_array, 256, 256)


def resize_to_128(nd_array: np.ndarray) -> np.ndarray:
    return resize_to(nd_array, 128, 128)


def resize_to(nd_array: np.ndarray, width: int, heigh: int) -> np.ndarray:
    im = Image.fromarray(nd_array)
    orig_width, orig_height = im.size
    min_size = min(orig_width, orig_height)
    left = int((orig_width - min_size) / 2)
    top = int((orig_height - min_size) / 2)
    right = int((orig_width + min_size) / 2)
    bottom = int((orig_height + min_size) / 2)
    im = im.crop((left, top, right, bottom))
    im = im.resize((width, heigh), Image.ANTIALIAS)
    return np.array(im, dtype=np.uint8)


def shift_to(orig_arr: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
    rez = np.zeros(orig_arr.shape)
    max_x = orig_arr.shape[0]
    max_y = orig_arr.shape[1]
    for x in range(0, max_x):
        for y in range(0, max_y):
            x_ = x + shift_x
            y_ = y + shift_y
            if x_ >= 0 and x_ < max_x and y_ >= 0 and y_ < max_y:
                rez[y + shift_y][x + shift_x] = orig_arr[y][x]
    return rez
