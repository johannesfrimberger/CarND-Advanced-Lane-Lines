import cv2
import numpy as np

"""
This file contains static helper functions to reduce the complexity of AdvancedLaneFinding class
"""

def region_of_interest(img, vertices, inverse = False):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    fill_color = 255

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (fill_color,) * channel_count
    else:
        ignore_mask_color = fill_color

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    if inverse:
        masked_image = cv2.bitwise_and(img, cv2.bitwise_not(mask))
    else:
        masked_image = cv2.bitwise_and(img, mask)

    return masked_image


def transform_to_bev(image):
    """
    Transform input image to birds eye view
    :param image: grayscale image
    :return: image in birds eye view
    """
    img_size = (image.shape[1], image.shape[0])
    src = np.float32([[600, 440], [800, 440], [1130, 650], [250, 660]])
    offset = 100
    dst = np.float32([[offset, offset], [img_size[0] - offset, offset],
                      [img_size[0] - offset, img_size[1] - offset],
                      [offset, img_size[1] - offset]])
    M = cv2.getPerspectiveTransform(src, dst)
    MInv = cv2.getPerspectiveTransform(src, dst)
    return [cv2.warpPerspective(image, M, img_size), MInv]