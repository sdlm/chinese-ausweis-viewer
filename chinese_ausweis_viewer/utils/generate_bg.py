import random
import numpy as np
from imgaug import augmenters as iaa
from imgaug import parameters as iap


def _admix_by_rand_mask_img(orig: np.ndarray, h_size=256, w_size=256, *, mask_pool):
    # prepare mask
    coef = max(iap.Normal(4, 1).draw_sample(), 0.5)
    mask = random.choice(mask_pool) / 255.0 * coef
    np.clip(mask, 0, 1, out=mask)

    # admix mono background
    addition = get_flat_background(h_size, w_size) / 255.0
    merge = (1.0 - mask) * orig / 255.0 + mask * addition

    #
    np.clip(merge, 0, 255, out=merge)
    return np.array(merge * 255.0, dtype=np.uint8)


def admix_by_rand_mask_img(orig: np.ndarray, iter_count: int = 1, *, mask_pool):
    for _ in range(iter_count):
        orig = _admix_by_rand_mask_img(orig, mask_pool=mask_pool)

    aug = iaa.Add(
        iap.Normal(0, 35),
        per_channel=1
    )
    orig = aug.augment_image(orig)
    np.clip(orig, 0, 255, out=orig)
    return orig


def generate_rand_bg(h_size=256, w_size=256, bg_admix_iters=5, *, mask_pool):
    back = get_flat_background(h_size, w_size)
    return admix_by_rand_mask_img(back, bg_admix_iters, mask_pool=mask_pool)


def get_flat_background(h_size=256, w_size=256):
    original_smpl = np.empty((h_size, w_size, 3), dtype=np.uint8)
    original_smpl.fill(128)
    aug = iaa.Add(
        iap.Normal(0, 45),
        per_channel=1
    )
    smpl = aug.augment_image(original_smpl)
    np.clip(smpl, 0, 255, out=smpl)
    return smpl


def get_mask(h_size=256, w_size=256):
    white_bg = np.empty((h_size, w_size, 3), dtype=np.uint8)
    white_bg.fill(255)
    aug = iaa.Sequential([
        iaa.CoarseDropout(
            iap.Positive(iap.Normal(0, 0.2)),
            size_percent=iap.Positive(iap.Normal(0, 0.1)) + 0.01
        ),
        iaa.GaussianBlur(
            sigma=iap.Positive(iap.Normal(0, 5)) + 1
        ),
        iaa.PiecewiseAffine(
            scale=iap.Positive(iap.Normal(0, 0.0333)) + 0.001
        )
    ])
    mask = aug.augment_image(white_bg)
    np.clip(mask, 0, 255, out=mask)
    return mask


def merge_by_mask(background, foreground, mask):
    w, h, chanells = background.shape
    merg_arr = np.array(background, dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if mask[i][j] > 100:
                for c in range(chanells):
                    merg_arr[i][j][c] = foreground[i][j][c]
    return merg_arr
