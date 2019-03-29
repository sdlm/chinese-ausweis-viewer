from datetime import date

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from .grab_fake_chinese_credentials import Person


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

    font_path = "data/wts11.ttf"
    in_font_path = "data/Inconsolata-Regular.ttf"
    inb_font_path = "data/Inconsolata-Bold.ttf"
    font = ImageFont.truetype(font_path, 76)

    line_len = 11
    address_lines = [
        person.address[i * line_len: (i+1) * line_len]
        for i in range(5)
        if bool(person.address[i * line_len: (i+1) * line_len])
    ]

    # name
    draw.text((1010, 560), person.name, (200, 0, 0), font=ImageFont.truetype(font_path, 90))
    # sex
    draw.text((1010, 720), person.sex, (200, 0, 0), font=font)
    # nationality
    draw.text((1460, 720), person.nationality, (200, 0, 0), font=font)
    # birthday
    draw.text((1515, 870), str(person.birthday.day), (200, 0, 0), font=ImageFont.truetype(inb_font_path, 82))
    draw.text((1330, 870), str(person.birthday.month), (200, 0, 0), font=ImageFont.truetype(inb_font_path, 82))
    draw.text((1040, 870), str(person.birthday.year), (200, 0, 0), font=ImageFont.truetype(inb_font_path, 82))
    # address
    for i, line in enumerate(address_lines):
        draw.text((1010, 1040 + i * 100), line, (200, 0, 0), font=font)
    # id
    for i, digit in enumerate(person.id):
        draw.text(
            (1340 + i * 62, 1420),
            digit,
            (200, 0, 0),
            font=ImageFont.truetype(inb_font_path, 100)
        )

    img.save('data/Chinese_card_constructor_test.jpg')
