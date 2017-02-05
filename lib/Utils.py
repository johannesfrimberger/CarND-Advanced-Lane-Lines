import cv2
import numpy as np
import matplotlib.image as mpimg
import os

"""
This file contains static helper functions to reduce the complexity of AdvancedLaneFinding class
"""


def region_of_interest(img, vertices, inverse=False):
    """

    :param img:
    :param vertices:
    :param inverse:
    :return:
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


def transform_to_bev(image, src, offset=(0, 0)):
    """
    Transform input image to birds eye view
    :param image: image that should be transformed
    :param src:
    :param offset:
    :return: image in birds eye view and inverse transformation matrix
    """
    img_size = (image.shape[1], image.shape[0])

    dst = np.float32([
        [offset[0], img_size[1] - offset[1]],
        [offset[0], offset[1]],
        [img_size[0] - offset[0], offset[1]],
        [img_size[0] - offset[0], img_size[1] - offset[1]]
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    MInv = cv2.getPerspectiveTransform(dst, src)

    return cv2.warpPerspective(image, M, img_size), MInv


def save_storage(store_results, storage_folder, file_name, identifier, image):
    """

    :param store_results:
    :param storage_folder:
    :param file_name:
    :param identifier:
    :param image:
    :return:
    """
    if store_results:
        output_file = os.path.join(storage_folder, identifier + os.path.basename(file_name))
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        imsize = image.shape
        if len(imsize) == 2:
            cv2.imwrite(output_file, cv2.cvtColor(image * 255, cv2.COLOR_GRAY2RGB))
        else:
            mpimg.imsave(output_file, image)


def draw_lanes(left_crv, right_crv, color_image, bev, MInv, radius, position):
    """

    :param left_crv:
    :param right_crv:
    :param bev:
    :return:
    """

    yvals = np.linspace(0, 100, num=101) * 7.2
    left_fitx = left_crv[0] * yvals ** 2 + left_crv[1] * yvals + left_crv[2]
    right_fitx = right_crv[0] * yvals ** 2 + right_crv[1] * yvals + right_crv[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(bev).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, MInv, (color_image.shape[1], color_image.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(color_image, 1, newwarp, 0.3, 0)

    #
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, 'Radius of curve = {:d} (m)'.format(int(radius)),
                (300, 50), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(result, 'Vehicle is {:.2f}m left of the center'.format(position),
                (300, 80), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    return result


def radius_and_position(left_lane, right_lane, y_eval, hor_center):
    """

    :param left_lane:
    :param right_lane:
    :param y_eval:
    :param hor_center:
    :return:
    """
    left_curverad = ((1 + (2 * left_lane[0] * y_eval + left_lane[1]) ** 2) ** 1.5) / np.absolute(2 * left_lane[0])
    right_curverad = ((1 + (2 * right_lane[0] * y_eval + right_lane[1]) ** 2) ** 1.5) / np.absolute(2 * right_lane[0])
    radius = (left_curverad + right_curverad) / 2.0

    left_position = left_lane[0] * y_eval ** 2 + left_lane[1] * y_eval + left_lane[2]
    right_position = right_lane[0] * y_eval ** 2 + right_lane[1] * y_eval + right_lane[2]
    scaling = 4.0 / (right_position - left_position)

    position = (hor_center - ((right_position - left_position) / 2.0) - left_position) * scaling

    return radius, position


def apply_mask(image, mask, extend=False, inverse=False):
    """
    Apply internally stored mask to image
    :param image: Image that should be masked
    :param mask: Mask that should be used
    :param extend: Add additional boundaries to mask
    :param inverse: Use inverse mask (everything else is kept except mask)
    :return: Masked image
    """
    mask_bl = mask[0]
    mask_tl = mask[1]
    mask_tr = mask[2]
    mask_br = mask[3]

    if extend:
        mask_bl = (mask_bl[0] - 50, mask_bl[1] + 50)
        mask_tl = (mask_tl[0] - 50, mask_tl[1] - 50)
        mask_tr = (mask_tr[0] + 50, mask_tr[1] - 50)
        mask_br = (mask_br[0] + 50, mask_br[1] + 50)

    vertices = np.array([[mask_bl, mask_tl, mask_tr, mask_br]],
                        dtype=np.int32)

    return region_of_interest(image, vertices, inverse)
