from typing import List

from random import randint
import numpy as np

import PIL
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from .grab_fake_chinese_credentials import get_chinese_creds


def get_card_generator(template_path: str, face_dir_path: str) -> PIL.Image:
    template = Image.open(template_path)
    batch_size = 35
    while True:
        creds = get_chinese_creds(batch_size)
        colors = get_batch_of_color(batch_size)
        for i, cred in enumerate(creds):
            template_with_creds = add_creds(template, cred, colors[i])
            yield add_face(template_with_creds, face_dir_path)


def get_batch_of_color(count: int) -> List[tuple]:
    r = g = b = np.absolute(np.random.normal(15, 5, count).astype(int))
    return list(zip(r, g, b))


def add_creds(template: PIL.Image, person: dict, color: tuple) -> PIL.Image:
    img = template.copy()
    draw = ImageDraw.Draw(img)

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

    return img


def add_face(img: PIL.Image, face_dir_path: str) -> PIL.Image:
    face_path = '{face_dir_path}{number:0>3}.png'.format(
        face_dir_path=face_dir_path,
        number=randint(1, 267)
    )
    face = Image.open(face_path)
    face = crop_img(face, 50)
    face = resize_to_width(face, 620)
    temp = Image.new('RGBA', img.size, 0)
    temp.paste(face, (1894, 623))
    return Image.alpha_composite(img, temp)


def crop_img(img: PIL.Image, value: int) -> PIL.Image:
    return img.crop(
        (value, value, img.size[0] - value, img.size[1] - value)
    )


def resize_to_width(img: PIL.Image, width: int) -> PIL.Image:
    height = int(img.size[1] * (620.0 / img.size[0]))
    return img.resize((width, height), PIL.Image.ANTIALIAS)
