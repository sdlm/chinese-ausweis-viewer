import os
import random
import time
from typing import Tuple

import imageio
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision.transforms import transforms
import math
import multiprocessing as mp

from .generate_bg import merge_by_mask
from .helpers import resize_to_128
from . import configs, card_generator as card_gen

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

CARD_POSITION = (649, 378 + 660)
CARD_SIZE = (2017, 1270)
MAX_ANGLE = 45
MIN_RATIO = 0.7
canvas_ratio = configs.IMAGE_SIZE / 3360.0
CARD_SIZE_256 = (CARD_SIZE[0] * canvas_ratio, CARD_SIZE[1] * canvas_ratio)


def get_sample_and_coords(original_mask, face_pool):
    card_generator = card_gen.get_card_generator(face_pool)
    mask, card, coords = get_sample_256(
        original_mask=np.copy(original_mask), card_generator=card_generator
    )
    gi_bg = get_gi_bg()
    card_with_bg = merge_by_mask(gi_bg, card, mask)
    to_tensor = transforms.ToTensor()
    return to_tensor(card_with_bg), [float(coords.item(k)) for k in range(8)]


class ChineseCardDataset(data.Dataset):

    def __init__(self, count: int):
        print('Initialize dataset')
        self.count = count

        start_time = time.time()
        to_tensor = transforms.ToTensor()
        self.values = [
            to_tensor(np.array(Image.open(f"data/trainset_128/{i:0>7}.png"), dtype=np.uint8))
            for i in range(count)
        ]
        # noinspection PyTypeChecker
        all_coords = np.loadtxt(open(f"data/trainset_128/coords.csv", "rb"), delimiter=",").astype(dtype=np.float32)
        self.labels = [all_coords[i] for i in range(count)]
        time_elapsed = time.time() - start_time
        print('samples loading elapsed: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        """
        Generates one sample of data

        return Tuple[value, label]
        value - torch.tensor - image
        label - List[Tuple[float]] - card corner's coords
        """
        return self.values[index], self.labels[index]


class ChineseCardClassificationDataset(data.Dataset):

    DIR = 'data/train/classification/128'

    def __init__(self, count: int):
        start_time = time.time()
        print('Initialize dataset')

        self.count = count
        to_tensor = transforms.ToTensor()
        self.values = [to_tensor(Image.open(f"{self.DIR}/{i:0>7}.png")) for i in range(count)]
        all_labels = np.loadtxt(f'{self.DIR}/labels.csv').astype(dtype=np.float)
        self.labels = [
            torch.from_numpy(array).long()
            for array in np.split(all_labels, all_labels.shape[0])
        ]
        time_elapsed = time.time() - start_time
        print('samples loading elapsed: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def __len__(self):
        return self.count

    def __getitem__(self, index) -> Tuple[torch.tensor, bool]:
        """
        Generates one sample of data

        return Tuple[value, label]
        value - torch.tensor - image
        label - bool - is card exists on image
        """
        # noinspection PyUnresolvedReferences
        return self.values[index], self.labels[index]


class NativeChineseCardDataset(data.Dataset):

    def __init__(self, count: int):
        print('Initialize dataset')
        self.count = count
        print('generate face pool')
        start_time = time.time()
        face_pool = card_gen.get_face_pool()
        time_elapsed = time.time() - start_time
        print('face pool generation elapsed: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        self.original_mask = card_gen.get_true_mask()
        self.card_generator = card_gen.get_card_generator(face_pool)
        self.to_tensor = transforms.ToTensor()

        print('generate samples dataset')
        start_time = time.time()
        self.values = []
        self.labels = []
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = [
                pool.apply_async(
                    get_sample_and_coords,
                    args=(self.original_mask, face_pool)
                )
                for _ in range(count)
            ]
            for x, y in [res.get() for res in results]:
                self.values.append(x)
                self.labels.append(y)
        time_elapsed = time.time() - start_time
        print('samples generation elapsed: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        for _ in range(count):
            mask, card, coords = get_sample_256(
                original_mask=np.copy(self.original_mask), card_generator=self.card_generator
            )
            gi_bg = get_gi_bg()
            card_with_bg = merge_by_mask(gi_bg, card, mask)
            self.values.append(self.to_tensor(card_with_bg))
            self.labels.append(coords)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        """
        Generates one sample of data

        return Tuple[value, label]
        value - torch.tensor - image
        label - List[Tuple[float]] - card corner's coords
        """
        return self.values[index], self.labels[index]


def get_sample_256(original_mask, card_generator):
    mask = original_mask
    card = card_generator.__next__()
    coords = get_orig_rectangle_coords(CARD_POSITION, CARD_SIZE)

    mask, card, coords = sample_to_256(mask, card, coords)

    theta = max(min(random.gauss(0, MAX_ANGLE / 3), MAX_ANGLE), -MAX_ANGLE)
    mask, card, coords = rotate_sample(mask, card, coords, theta)

    max_ratio = mask.shape[0] / math.sqrt(CARD_SIZE_256[0] ** 2 + CARD_SIZE_256[1] ** 2)
    ratio = random.uniform(MIN_RATIO, max_ratio)
    mask, card, coords = resize_sample(mask, card, coords, ratio)

    shift_x, shift_y = get_shift(coords, mask.shape)
    mask, card, coords = shift_sample(mask, card, coords, shift_x, shift_y)

    return mask, card, coords


def rotate_sample(mask, card, coords, theta):
    new_coords = rotate_coords(coords, theta, mask.shape)
    new_card = rotate_image(card, theta)
    new_mask = rotate_image(mask, theta)
    return new_mask, new_card, new_coords


def resize_sample(mask, card, coords, ratio):
    new_coords = resize_coords(coords, ratio, mask.shape)
    new_card = resize_image(card, ratio)
    new_mask = resize_image(mask, ratio)
    return new_mask, new_card, new_coords


def shift_sample(mask, card, coords, shift_x, shift_y):
    new_coords = shift_coords(coords, shift_x, shift_y)
    new_card = shift_image(card, shift_x, shift_y)
    new_mask = shift_image(mask, shift_x, shift_y)
    return new_mask, new_card, new_coords


def get_gi_bg(path: str = './data/images', index: int = None):
    filenames = os.listdir(path)
    while True:
        if index is None:
            filename = random.choice(filenames)
        else:
            filename = filenames[index]
        try:
            arr = load_gi_bg(path, filename)
        except:
            continue
        else:
            return resize_to_128(arr)


def load_gi_bg(path, filename):
    file_path = os.path.join(path, filename)
    img = imageio.imread(file_path, pilmode="RGB")
    return np.array(img, dtype=np.uint8)


def get_orig_rectangle_coords(position, size) -> np.array:
    return np.array([
        [position[0], position[1]],
        [position[0] + size[0], position[1]],
        [position[0], position[1] + size[1]],
        [position[0] + size[0], position[1] + size[1]],
    ])


def sample_to_256(mask, card, coords):
    ratio = 128.0 / mask.shape[0]
    new_coords = resize_coords(coords, ratio, mask.shape, with_canvas=True)
    new_card = resize_to_128(card)
    new_mask = resize_to_128(mask)
    return new_mask, new_card, new_coords


def rotate_coords(coords, angle, img_shape):
    a = angle / 180.0 * np.pi
    rotation_matrix = np.matrix(((np.cos(a), -np.sin(a)), (np.sin(a), np.cos(a))))
    x_center = img_shape[0]/2
    y_center = img_shape[1]/2
    coords_ = np.copy(coords)
    coords_[:, 0] = coords_[:, 0] - x_center
    coords_[:, 1] = coords_[:, 1] - y_center
    new_coords = coords_ @ rotation_matrix
    new_coords[:, 0] = new_coords[:, 0] + x_center
    new_coords[:, 1] = new_coords[:, 1] + y_center
    return new_coords


def rotate_image(img_arr: np.array, angle) -> np.array:
    layers_map = {
        1: 'L',
        3: 'RGB',
        4: 'RGBA'
    }
    layer_count = img_arr.shape[2] if len(img_arr.shape) == 3 else 1
    image = Image.fromarray(img_arr.astype('uint8'), layers_map[layer_count])
    image = image.rotate(angle, resample=Image.BILINEAR, expand=False)
    return np.array(image)


def resize_coords(coords, ratio, img_shape, with_canvas: bool = False):
    x_center = img_shape[0]/2
    y_center = img_shape[1]/2
    coords_ = np.copy(coords)
    coords_[:, 0] = coords_[:, 0] - x_center
    coords_[:, 1] = coords_[:, 1] - y_center
    new_coords = np.multiply(coords_, ratio)
    if with_canvas:
        x_center *= ratio
        y_center *= ratio
    new_coords[:, 0] = new_coords[:, 0] + x_center
    new_coords[:, 1] = new_coords[:, 1] + y_center
    return new_coords


def resize_image(img_arr: np.array, ratio) -> np.array:
    layers_map = {
        1: 'L',
        3: 'RGB',
        4: 'RGBA'
    }
    layer_count = img_arr.shape[2] if len(img_arr.shape) == 3 else 1
    image = Image.fromarray(img_arr.astype('uint8'), layers_map[layer_count])
    orig_height = image.height
    orig_width = image.width
    new_height = int(image.height * ratio)
    new_width = int(image.width * ratio)
    image = image.resize((new_width, new_height))
    if new_width > orig_width:
        left = int((new_width - orig_width) / 2)
        top = int((new_height - orig_height) / 2)
        right = left + orig_width
        bottom = top + orig_height
        image = image.crop((left, top, right, bottom))
        return np.array(image)
    else:
        left = int((orig_width - new_width) / 2)
        top = int((orig_height - new_height) / 2)
        new_image = Image.new(layers_map[layer_count], (orig_width, orig_height))
        new_image.paste(image, (left, top))
        return np.array(new_image)


def get_shift(coords, original_shape):
    left_gap = np.floor(np.amin(coords, axis=0))
    right_gap = np.array(original_shape) - np.ceil(np.amax(coords, axis=0))
    max_shift_x = min(left_gap.item(0), right_gap.item(0)) - original_shape[0] * 0.001
    max_shift_y = min(left_gap.item(1), right_gap.item(1)) - original_shape[0] * 0.001
    shift_x = max(min(random.gauss(0, max_shift_x / 3), max_shift_x), -max_shift_x)
    shift_y = max(min(random.gauss(0, max_shift_y / 3), max_shift_y), -max_shift_y)
    return shift_x, shift_y


def shift_coords(coords, shift_x, shift_y):
    new_coords = np.copy(coords)
    new_coords[:, 0] = new_coords[:, 0] + shift_x
    new_coords[:, 1] = new_coords[:, 1] + shift_y
    return new_coords


def shift_image(img_arr: np.array, shift_x, shift_y) -> np.array:
    layers_map = {
        1: 'L',
        3: 'RGB',
        4: 'RGBA'
    }
    layer_count = img_arr.shape[2] if len(img_arr.shape) == 3 else 1
    image = Image.fromarray(img_arr.astype('uint8'), layers_map[layer_count])
    image = image.transform(image.size, Image.AFFINE, (1, 0, -shift_x, 0, 1, -shift_y))
    return np.array(image)
