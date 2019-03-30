from random import randint
import numpy as np
from imgaug import augmenters as iaa
from imgaug import parameters as iap

MEAN_FLAT_BG = 128
VARIANCE_FLAT_BG = MEAN_FLAT_BG * (2/3)  # because we have 3 dimension RGB

ADMIX_FLAT_BG_BY_MASK_AUG = iaa.FrequencyNoiseAlpha(
    exponent=iap.Uniform(-4, 4),
    first=iaa.Add(255),
    size_px_max=32,
    iterations=1,
    sigmoid=True,
    sigmoid_thresh=(-10, 10)
)


def generate_rand_bg(h_size=256, w_size=256, bg_admix_iters=5):
    smpl = get_flat_background_with_relative_colors_uniform_distribution(h_size, w_size)
    
    for _ in range(randint(1, bg_admix_iters)):
        smpl = admix_flat_bg_by_mask(smpl)
    
    np.clip(smpl * 255, 0, 255, out=smpl)
    return smpl.astype(np.uint8)


def admix_flat_bg_by_mask(orig: np.ndarray) -> np.ndarray:
    img_for_adimx = get_flat_background_with_relative_colors_uniform_distribution(
        h_size=orig.shape[0], 
        w_size=orig.shape[1]
    )
    black = np.zeros(orig.shape, dtype=np.uint8)
    mask = ADMIX_FLAT_BG_BY_MASK_AUG.augment_image(black) / 255.0
    blended = (1.0 - mask) * orig + mask * img_for_adimx  # like PIL.Image.composite()
    np.clip(blended, 0, 1, out=blended)
    return blended


def get_flat_background_with_relative_colors_uniform_distribution(
    h_size: int = 256, 
    w_size: int = 256
) -> np.ndarray:
    return get_flat_background_uniform_distribution(h_size, w_size).astype('float32') / 255.0


def get_flat_background_normal_distribution(
    h_size: int = 256, 
    w_size: int = 256, *, 
    mean: float = MEAN_FLAT_BG, 
    variance: float = VARIANCE_FLAT_BG
) -> np.ndarray:
    colors = np.random.normal(mean, variance, 3).astype(np.uint8)
    return _get_flat_background(h_size, w_size, colors)


def get_flat_background_uniform_distribution(
    h_size: int = 256, 
    w_size: int = 256, *, 
    min_val: float = 0, 
    max_val: float = 255
) -> np.ndarray:
    colors = np.random.uniform(min_val, max_val, 3).astype(np.uint8)
    return _get_flat_background(h_size, w_size, colors)


def _get_flat_background(h_size, w_size, colors):
    np.clip(colors, 0, 255, out=colors)
    r = np.full((h_size, w_size), colors[0])
    g = np.full((h_size, w_size), colors[1])
    b = np.full((h_size, w_size), colors[2])
    return np.dstack([r,g,b])


def merge_by_mask(background, foreground, mask):
    w, h, chanells = background.shape
    merg_arr = np.array(background, dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if mask[i][j] > 100:
                for c in range(chanells):
                    merg_arr[i][j][c] = foreground[i][j][c]
    return merg_arr


get_flat_background = get_flat_background_normal_distribution
