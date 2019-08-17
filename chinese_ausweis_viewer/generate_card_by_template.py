from dataclasses import dataclass
from datetime import date

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

# from .grab_fake_chinese_credentials import Person


@dataclass
class Person:
    name: str
    sex: str
    nationality: str
    birthday: date
    address: str
    id: str


if __name__ == '__main__':

    person = Person(
        name='简芳',
        sex='女',
        nationality='汉',
        birthday=date(1991, 12, 22),
        address='新疆维吾尔族自治区巴音郭楞蒙古自治州和静县',
        id='652827199105021201'
    )

    img = Image.open("data/Chinese_card_constructor.jpg")
    draw = ImageDraw.Draw(img)

    line_len = 11
    address_lines = [
        person.address[i * line_len: (i+1) * line_len]
        for i in range(5)
        if bool(person.address[i * line_len: (i+1) * line_len])
    ]

    # brush configs
    color = (200, 0, 0)
    # font_path = "data/wts11.ttf"
    font_path = "data/MSHEI.ttf"
    in_font_path = "data/Inconsolata-Regular.ttf"
    inb_font_path = "data/Inconsolata-Bold.ttf"
    font = ImageFont.truetype(font_path, 76)
    name_font = ImageFont.truetype(font_path, 90)
    birthday_font = ImageFont.truetype(inb_font_path, 82)

    # name
    draw.text((1010, 560), person.name, color, font=name_font)

    # sex
    draw.text((1010, 720), person.sex, color, font=font)

    # nationality
    draw.text((1430, 721), person.nationality, color, font=font)

    # birthday
    y_birthday = 865
    draw.text((1040, y_birthday), str(person.birthday.year), color, font=birthday_font)
    month_str = str(person.birthday.month)
    x_month = 1330 if len(month_str) == 2 else 1330 + 30
    draw.text((x_month, y_birthday), month_str, color, font=birthday_font)
    day_str = str(person.birthday.day)
    x_day = 1520 if len(day_str) == 2 else 1520 + 30
    draw.text((x_day, y_birthday), day_str, color, font=birthday_font)

    # address
    for i, line in enumerate(address_lines):
        draw.text((1010, 1040 + i * 100), line, color, font=font)

    # id
    for i, digit in enumerate(person.id):
        draw.text(
            (1340 + i * 62, 1420),
            digit,
            color,
            font=ImageFont.truetype(inb_font_path, 100)
        )

    img.save('data/Chinese_card_constructor_test.jpg')
