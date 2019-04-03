from typing import List, Generator

from random import randint, choice
import numpy as np

import PIL
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from .grab_fake_chinese_credentials import get_chinese_creds
from . import configs

FACES_COUNT = 267


def get_true_mask() -> np.ndarray:
    mask = Image.open(configs.TRUE_MASK_PATH)
    canvas = Image.new('RGB', (3360, 3360), (0, 0, 0))
    canvas.paste(mask, box=(0, 660, mask.size[0], mask.size[1] + 660))
    return np.array(canvas.convert('L'), dtype=np.uint8)


def get_card_generator(
        template_path: str = configs.CARD_TEMPLATE_PATH,
        face_dir_path: str = configs.FACE_DIR_PATH
) -> Generator[np.ndarray, None, None]:
    card_with_face_pool = get_card_with_face_pool(template_path, face_dir_path)
    batch_size = 35
    while True:
        creds = get_chinese_creds(batch_size)
        colors = get_batch_of_color(batch_size)
        for i, cred in enumerate(creds):
            card_with_face = choice(card_with_face_pool).copy()
            complete_card = add_creds(card_with_face, cred, colors[i])
            card_canvas = Image.new('RGBA', (3360, 3360), (0, 0, 0, 0))
            card_canvas.paste(
                complete_card,
                box=(0, 660, complete_card.size[0], complete_card.size[1] + 660)
            )
            yield np.array(card_canvas)


def get_batch_of_color(count: int) -> List[tuple]:
    r = g = b = np.absolute(np.random.normal(15, 5, count).astype(int))
    return list(zip(r, g, b))


def add_creds(template: PIL.Image.Image, person: dict, color: tuple) -> PIL.Image.Image:
    draw = ImageDraw.Draw(template)

    line_len = 11
    address_lines = [
        person['address'][i * line_len: (i + 1) * line_len]
        for i in range(5)
        if bool(person['address'][i * line_len: (i + 1) * line_len])
    ]

    # brush configs
    font_path = "data/MSHEI.ttf"
    inconsolata_font_path = "data/Inconsolata-Bold.ttf"
    font = ImageFont.truetype(font_path, 76)
    name_font = ImageFont.truetype(font_path, 90)
    birthday_font = ImageFont.truetype(inconsolata_font_path, 82)

    # name
    draw.text((1010, 560), person['name'], color, font=name_font)

    # sex
    draw.text((1010, 720), person['sex'], color, font=font)

    # nationality
    draw.text((1430, 721), person['nationality'], color, font=font)

    # birthday
    y_birthday = 865
    draw.text((1040, y_birthday), str(person['birthday'].year), color, font=birthday_font)
    month_str = str(person['birthday'].month)
    x_month = 1330 if len(month_str) == 2 else 1330 + 30
    draw.text((x_month, y_birthday), month_str, color, font=birthday_font)
    day_str = str(person['birthday'].day)
    x_day = 1520 if len(day_str) == 2 else 1520 + 30
    draw.text((x_day, y_birthday), day_str, color, font=birthday_font)

    # address
    for i, line in enumerate(address_lines):
        draw.text((1010, 1040 + i * 100), line, color, font=font)

    # id
    for i, digit in enumerate(person['id']):
        draw.text(
            (1340 + i * 62, 1420),
            digit,
            color,
            font=ImageFont.truetype(inconsolata_font_path, 100)
        )

    return template


def get_card_with_face_pool(
        template_path: str = configs.CARD_TEMPLATE_PATH,
        face_dir_path: str = configs.FACE_DIR_PATH,
) -> List[PIL.Image.Image]:
    template = Image.open(template_path)
    face_pool = []
    for face_number in range(1, FACES_COUNT + 1):
        template_ = template.copy()
        template_ = add_face(template_, face_dir_path, face_number=face_number)
        face_pool.append(template_)
    return face_pool


def add_face(img: PIL.Image.Image, face_dir_path: str, *, face_number: int = None) -> PIL.Image.Image:
    face_path = '{face_dir_path}{number:0>3}.png'.format(
        face_dir_path=face_dir_path,
        number=face_number or randint(1, FACES_COUNT)
    )
    face = Image.open(face_path)
    face = crop_img(face, 50)
    face = resize_to_width(face, 620)
    temp = Image.new('RGBA', img.size, 0)
    temp.paste(face, (1894, 623))
    return Image.alpha_composite(img, temp)


def crop_img(img: PIL.Image.Image, value: int) -> PIL.Image.Image:
    return img.crop(
        (value, value, img.size[0] - value, img.size[1] - value)
    )


def resize_to_width(img: PIL.Image.Image, width: int) -> PIL.Image.Image:
    height = int(img.size[1] * (620.0 / img.size[0]))
    return img.resize((width, height), PIL.Image.ANTIALIAS)
