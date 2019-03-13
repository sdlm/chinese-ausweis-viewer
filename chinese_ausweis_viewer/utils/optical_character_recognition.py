import numpy as np

from imgaug import augmenters as iaa
from PIL import Image
import pytesseract

LABELS = [
    'name_label',
    'sex_label',
    'nationality_label',
    'birthday_label',
    'year_label',
    'month_label',
    'day_label',
    'address_label',
    'id_number_label',
    'name_value',
    'sex_value',
    'nationality_value',
    'year_value',
    'month_value',
    'day_value',
    'address_value',
    'id_number_value',
]


def extract_data(cropped_card, box_coords):
    image_data = {}
    for i, box in enumerate(box_coords):
        title = LABELS[i]
        img_part = crop_img_part_by_box(cropped_card, box)
        gray_img_part = grayscale(img_part)
        image_data[title] = ''
        k = 6
        while k > 0:
            k -= 1
            text = extract_text(gray_img_part)
            if text:
                image_data[title] = text
                break
            else:
                gray_img_part = prepare_img_part(gray_img_part)
    return image_data


def get_actual_box_coords(boxes, cropped_card):
    coefficient = np.array([float(cropped_card.shape[1]), float(cropped_card.shape[0])])
    return [
        np.int0(box * coefficient)
        for box in boxes
    ]


def crop_img_part_by_box(img, box):
    a1, a2 = box.min(axis=0), box.max(axis=0)
    return img[a1[1]:a2[1], a1[0]:a2[0]]


def extract_text(img):
    pillow_img = Image.fromarray(img)
    return pytesseract.image_to_string(pillow_img, lang='chi_sim', config='--psm 1 --oem 1')


def grayscale(img):
    return iaa.Grayscale(alpha=1.0).augment_image(img)


def prepare_img_part(img):
    """
    Use some magic coefficients/ like 1.05
    """
    prepare_seq = iaa.Sequential([
        iaa.Multiply(1.05),
        iaa.ContrastNormalization(1.05),
        iaa.Sharpen(alpha=0.04, lightness=1.05)
    ])
    return prepare_seq.augment_image(img)
