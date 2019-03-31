import imageio
import numpy as np
from imgaug import augmenters as iaa
from imgaug import parameters as iap

from .generate_flat_bg import get_flat_simple_bg_generator, get_simple_bg_generator
from .generate_bg import get_rand_bg_generator, merge_by_mask
from .helpers import resize_to_256
from . import configs


affine_aug = iaa.Affine(
    scale=iap.Clip(iap.Normal(1, 0.15), 0.7, 1.3),
    rotate=iap.Normal(0, 6),
    shear=iap.Normal(0, 6),
    mode='wrap'
)

crop_aug = iaa.Sometimes(
    0.9,
    iaa.Crop(
        percent=iap.Clip(iap.Positive(iap.Normal(0, 0.15)), 0, 0.25),
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
            (0, 0.2),
            size_percent=(0.02, 0.3),
            per_channel=0.5
        ),
    ),

])


def get_next_batch(
    batch_size,
    w_size: int = configs.W_SIZE,
    h_size: int = configs.H_SIZE,
):
    # load original images
    sample_img = imageio.imread(configs.ORIGINAL_SMPL_PATH, pilmode="RGB")
    original_smpl = np.array(sample_img, dtype=np.uint8)

    mask_img = imageio.imread(configs.ORIGINAL_MASK_PATH, pilmode="RGB", as_gray=True)
    original_mask = np.array(mask_img, dtype=np.uint8)

    # make train set
    train_x = np.empty((batch_size, w_size, h_size, 3), dtype='float32')
    train_y = np.empty((batch_size, w_size, h_size, 1), dtype='float32')

    # make generators
    bg_generator = get_bg_generator()
    next_pair_foo = next_pair_generator(original_mask, original_smpl, bg_generator)

    # fill train set
    for i in range(batch_size):
        img, msk = next_pair_foo.__next__()
        train_x[i] = img
        train_y[i] = msk

    return train_x, train_y


def next_pair_generator(original_mask, original_smpl, bg_generator):
    while True:
        _affine_aug = affine_aug._to_deterministic()
        _crop_aug = crop_aug._to_deterministic()

        # prepare mask
        _mask = _affine_aug.augment_image(original_mask)
        _mask = _crop_aug.augment_image(_mask)
        _mask = resize_to_256(_mask)

        # prepare card
        _smpl = _affine_aug.augment_image(original_smpl)
        _smpl = _crop_aug.augment_image(_smpl)
        _smpl = resize_to_256(_smpl)

        # get background
        _bckg = bg_generator.__next__()

        # composite background and card
        _smpl = merge_by_mask(_bckg, _smpl, _mask)

        # make most strong augmentation
        _smpl = random_aug.augment_image(_smpl)

        _mask = _mask / 255.0
        _smpl = _smpl / 255.0

        _mask = _mask.reshape(configs.W_SIZE, configs.H_SIZE, 1)
        yield _smpl, _mask


def get_bg_generator():
    rand_bg_generator = get_rand_bg_generator()
    flat_simple_bg_generator = get_flat_simple_bg_generator()
    simple_bg_generator = get_simple_bg_generator()
    while True:
        picker = np.random.randint(low=1, high=100)
        if picker < 60:
            yield rand_bg_generator.__next__()
        elif picker < 80:
            yield flat_simple_bg_generator.__next__()
        else:
            yield simple_bg_generator.__next__()
