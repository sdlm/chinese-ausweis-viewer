import os
import random
from typing import Generator
import numpy as np

import imageio
from imgaug import augmenters as iaa
from imgaug import parameters as iap

from .helpers import resize_to_256
from . import configs

aug_for_flat_simple_bg = iaa.Add(
    iap.Normal(0, 10),
    per_channel=0.8
)

aug_for_simple_bg = iaa.Sequential([
    iaa.Add(
        iap.Normal(0, 10),
        per_channel=0.8
    ),
    iaa.ContrastNormalization(
        iap.Normal(1, 0.1),
        per_channel=0.8
    ),
    iaa.Pad(
        percent=iap.Positive(iap.Normal(0, 0.2)),
        pad_mode=['wrap']
    ),
    iaa.Affine(
        scale=iap.Positive(iap.Normal(1, 0.1)),
        rotate=iap.Normal(0, 15),
        shear=iap.Normal(0, 5),
        mode='wrap'
    )
])

aug_for_gi_images = iaa.Sequential([
    iaa.Add(
        iap.Normal(0, 50),
        per_channel=0.3
    ),
    iaa.ContrastNormalization(
        iap.Normal(1, 0.1),
        per_channel=0.3
    ),
    iaa.Affine(
        scale=iap.Normal(1, 0.1),
        rotate=iap.Normal(0, 20),
        shear=iap.Normal(0, 5),
        mode='wrap'
    )
])


def get_flat_simple_bg_generator() -> Generator[np.ndarray, None, None]:
    simplest_flat_bg = imageio.imread(configs.SIMPLEST_FLAT_BG_PATH, pilmode='RGB')
    while True:
        pic = aug_for_flat_simple_bg.augment_image(simplest_flat_bg)
        yield np.array(pic)


def get_simple_bg_generator() -> Generator[np.ndarray, None, None]:
    simple_bg = imageio.imread(configs.SIMPLEST_BG_PATH, pilmode='RGB')
    while True:
        pic = aug_for_simple_bg.augment_image(simple_bg)
        yield np.array(pic)


def get_bg_from_gi_generator() -> Generator[np.ndarray, None, None]:
    path_list = [
        os.path.join(configs.GI_IMAGES_PATH, filename)
        for filename in os.listdir(configs.GI_IMAGES_PATH)
    ]
    while True:
        img_path = random.choice(path_list)
        pic = imageio.imread(img_path, pilmode='RGB')
        pic = resize_to_256(pic)
        pic = aug_for_gi_images.augment_image(pic)
        yield np.array(pic)
