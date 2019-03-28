from dataclasses import dataclass
from datetime import date

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


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
        birthday=date(1991, 5, 2),
        address='新疆维吾尔族自治区巴音郭楞蒙古自治州和静县',
        id='652827199105021201'
    )

    img = Image.open("data/Chinese_card_constructor.jpg")
    draw = ImageDraw.Draw(img)

    font_path = "data/wts11.ttf"
    font = ImageFont.truetype(font_path, 76)

    # name
    draw.text((1010, 580), person.name, (200, 0, 0), font=font)
    # sex
    draw.text((1010, 730), person.sex, (200, 0, 0), font=font)
    # nationality
    draw.text((1460, 730), person.nationality, (200, 0, 0), font=font)
    # birthday
    draw.text((1525, 880), str(person.birthday.day), (200, 0, 0), font=font)
    draw.text((1340, 880), str(person.birthday.month), (200, 0, 0), font=font)
    draw.text((1010, 880), str(person.birthday.year), (200, 0, 0), font=font)
    # address
    draw.text((1010, 1060), person.address, (200, 0, 0), font=font)
    # id
    draw.text((1340, 1460), person.id, (200, 0, 0), font=font)

    img.save('data/Chinese_card_constructor_test.jpg')
