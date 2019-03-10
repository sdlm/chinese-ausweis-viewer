import numpy as np
from imgaug import augmenters as iaa
from imgaug import parameters as iap

from generate_flat_bg import get_flat_simple_bg, get_simple_bg
from generate_bg import generate_rand_bg, merge_by_mask
from helpers import resize_to_256


affine_aug = iaa.Affine(
    scale=iap.Positive(iap.Normal(1, 0.2)),
    rotate=iap.Normal(0, 6),
    shear=iap.Normal(0, 6),
    mode='wrap'
)

crop_aug = iaa.Sometimes(
    0.9,
    iaa.Crop(
        percent=iap.Positive(iap.Normal(0, 0.1)),
        sample_independently=True,
        keep_size=False
    ),
)

random_aug = iaa.Sequential([

    iaa.Sometimes(
        0.5,
        iaa.OneOf([
            # Small blur
            iaa.OneOf([
                iaa.GaussianBlur(sigma=(0, 0.8)),
                iaa.AverageBlur(k=(0, 2)),
                iaa.MedianBlur(k=(1, 3)),
            ]),

            # Strengthen or weaken the contrast in each image.
            iaa.ContrastNormalization((0.9, 1.1)),

            # Add gaussian noise.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.3),
        ])
    ),

    iaa.Sometimes(
        0.8,
        # Make some images brighter and some darker.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
    ),

    iaa.Sometimes(
        0.4,
        # Augmenter that sets rectangular areas within images to zero.
        iaa.CoarseDropout(
            (0, 0.1),
            size_percent=(0.02, 0.3),
            per_channel=0.5
        ),
    ),

])


def next_pair(
    original_mask,
    original_smpl,
    simplest_flat_bg,
    simple_bg,
    mask_pool,
):
    _affine_aug = affine_aug._to_deterministic()
    _crop_aug = crop_aug._to_deterministic()

    _mask = _affine_aug.augment_image(original_mask)
    _mask = _crop_aug.augment_image(_mask)
    _mask = resize_to_256(_mask)

    _smpl = _affine_aug.augment_image(original_smpl)
    _smpl = _crop_aug.augment_image(_smpl)
    _smpl = resize_to_256(_smpl)

    _smpl = iaa.Add(iap.Normal(0, 10)).augment_image(_smpl)

    rand_state = np.random.randint(low=1, high=100)
    if rand_state < 40:
        _bckg = generate_rand_bg(mask_pool=mask_pool)
    elif rand_state < 70:
        _bckg = get_flat_simple_bg(simplest_flat_bg)
    else:
        _bckg = get_simple_bg(simple_bg)

    _smpl = merge_by_mask(_bckg, _smpl, _mask)
    _smpl = random_aug.augment_image(_smpl)

    _mask = _mask / 255.0
    _smpl = _smpl / 255.0

    _mask = _mask.reshape(256, 256, 1)
    return _smpl, _mask
