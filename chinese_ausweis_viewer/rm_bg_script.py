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
        new_img.save(to_)
        j += 1
