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


def resize_to_256(nd_array):
    im = Image.fromarray(nd_array)
    width, height = im.size
    min_size = min(width, height)
    left = int((width - min_size) / 2)
    top = int((height - min_size) / 2)
    right = int((width + min_size) / 2)
    bottom = int((height + min_size) / 2)
    im = im.crop((left, top, right, bottom))
    im = im.resize((256, 256), Image.ANTIALIAS)
    return np.array(im, dtype=np.uint8)
