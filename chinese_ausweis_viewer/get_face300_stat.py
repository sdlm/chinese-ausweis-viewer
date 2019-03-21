import numpy as np
from PIL import Image


if __name__ == '__main__':

    shapes = set()

    j = 1
    for i in range(320):
        from_ = f'data/face300/{i:0>3}.jpg'
        try:
            pic = Image.open(from_)
        except FileNotFoundError:
            continue
        arr = np.array(pic)
        if arr.shape not in shapes:
            shapes.add(arr.shape)

    print(shapes)
