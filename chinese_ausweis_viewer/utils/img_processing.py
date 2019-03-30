from typing import Tuple

import numpy as np
import cv2 as cv


def remove_bg(img: np.ndarray, iterations_count: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    # Now there will be some regions in the original image where you are simply sure, 
    # that part belong to bg. Mark such region with 255 in marker image. 
    # Now the region where you are sure to be the fg are marked with 128. 
    # The region you are not sure are marked with 0.
    __sure = 255
    __not_sure = 0
    __sure_not = 128

    # get bg mask by threshold (like all pixels lighter than ..)
    gray = cv.cvtColor(img.copy(), cv.COLOR_RGB2GRAY)
    _, trash = cv.threshold(gray, 0, __sure, cv.THRESH_TRIANGLE)

    # reduce bg area, for remove some small areas in face/clothe
    bg_sure = cv.erode(trash, None, iterations=iterations_count)

    # increase foreground area
    fg_sure = cv.dilate(trash, None, iterations=iterations_count)

    # mark fg with 128 value
    _, fg = cv.threshold(fg_sure, 1, __sure_not, 1)

    # combine them to one mask with markers: 255, 128, 0
    marker = cv.add(fg, bg_sure)

    marker32 = np.int32(marker)
    cv.watershed(img, marker32)
    mask = cv.convertScaleAbs(marker32)

    mask = _switch_bg_and_fg(mask)

    img_copy = img.copy()
    _, trash = cv.threshold(mask, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    result = cv.bitwise_and(img_copy, img_copy, mask=trash)

    return result, trash


def _get_path_with_ext_png(path: str) -> str:
    return ''.join(path.rsplit('.', 1)[:-1]) + '.png'


def _switch_bg_and_fg(mask: np.ndarray) -> np.ndarray:
    __temp_val = 123
    __true = 255
    __false = 128
    mask[mask == __true] = __temp_val
    mask[mask == __false] = __true
    mask[mask == __temp_val] = __false
    return mask
