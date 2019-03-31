import imageio
from imgaug import augmenters as iaa
from imgaug import parameters as iap

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


def get_flat_simple_bg_generator():
    simplest_flat_bg = imageio.imread(configs.SIMPLEST_FLAT_BG_PATH, pilmode='RGB')
    while True:
        yield aug_for_flat_simple_bg.augment_image(simplest_flat_bg)


def get_simple_bg_generator():
    simple_bg = imageio.imread(configs.SIMPLEST_BG_PATH, pilmode='RGB')
    while True:
        yield aug_for_simple_bg.augment_image(simple_bg)
