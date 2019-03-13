import numpy as np
import cv2

from imgaug import augmenters as iaa


def crop_by_mask(mask_256, orig_img):
    # rotate orig img
    angle = get_angle_of_bound_box(mask_256)
    reversed_mask_256 = rotate_mask_on_angle(mask_256, angle)
    reversed_card = rotate_mask_on_angle(orig_img, angle)

    # find bound box
    box = get_bound_box(reversed_mask_256)
    orig_size_box = box * (orig_img.shape[0] / 256.0)
    orig_size_box_int = np.int0(orig_size_box)

    # crop
    a1, a2 = orig_size_box_int.min(axis=0), orig_size_box_int.max(axis=0)
    cropped_card = reversed_card[a1[1]:a2[1], a1[0]:a2[0]]

    return cropped_card


def get_bound_box(mask):
    cnt = get_contour(mask)
    rect = cv2.minAreaRect(cnt)
    return cv2.boxPoints(rect)


def get_angle_of_bound_box(mask):
    cnt = get_contour(mask)
    rect = cv2.minAreaRect(cnt)
    temp_angle = rect[2]
    angle = 90 + temp_angle if abs(temp_angle) > 45 else temp_angle
    return angle


def get_contour(mask):
    _, threshold = cv2.threshold(mask, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]


def rotate_mask_on_angle(mask, angle):
    affine_aug = iaa.Affine(rotate=-angle, mode='wrap')
    return affine_aug.augment_image(mask)
