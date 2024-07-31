import cv2
import numpy as np


def narrowing_mandibular_canal(right_M3_channel_pixels, left_M3_channel_pixels):
    # First, max_contour is found by max(contours, key=cv2.contourArea)
    # Second , The top and bottom of the canal contours point are found by y_coordinates = [point[0][1] for point in max_contour]
    # Third , The difference of the y coordinates are calculated as max and mean for calculating ratio of narrowing
    def ratio_narrowing(img):
        img[img > 0] = 1
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(img, max_contour, -1, (255, 255, 255), 2)

        y_coordinates = [point[0][1] for point in max_contour]

        min_width = min(y_coordinates)
        max_width = max(y_coordinates)

        ratio_of_narrowing = (max_width - min_width) / (max_width)

        return ratio_of_narrowing

    if np.array_equal(right_M3_channel_pixels, []):
        rigt_m3_narrowing = -1
    else:
        rigt_m3_narrowing = ratio_narrowing(right_M3_channel_pixels)
    if np.array_equal(left_M3_channel_pixels, []):
        left_m3_narrowing = -1
    else:
        left_m3_narrowing = ratio_narrowing(left_M3_channel_pixels)

    return rigt_m3_narrowing, left_m3_narrowing
