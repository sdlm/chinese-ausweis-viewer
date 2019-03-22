import numpy as np
from PIL import Image

from .utils.img_processing import remove_bg


if __name__ == '__main__':

    j = 1
    for i in range(320):
        from_ = f'data/face300/{i:0>3}.jpg'
        try:
            pic = Image.open(from_)
        except FileNotFoundError:
            continue
        arr = np.array(pic)
        result = remove_bg(arr)
        new_img = Image.fromarray(result)
        to_ = f'data/face300_mod1/{j:0>3}.jpg'

        w, h = new_img.size
        new_w, new_h = 682, 768
        half_crop_w = (w - new_w) / 2
        new_img_crop = new_img.crop((half_crop_w, 0, w - half_crop_w, new_h))

        new_img_crop.save(to_)
        j += 1
