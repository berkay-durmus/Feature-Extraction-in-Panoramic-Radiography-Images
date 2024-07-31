import cv2
import numpy as np


def discontinuity_canal(right_M3_pixels, right_M3_channel_pixels, left_M3_pixels, left_M3_channel_pixels):
    # This part finds the discontinuity canal
    # Firts, we are binarized the all images
    # Second, The contours is found by cv2.findContours()
    # Third, we set the pixels where the tooth and the channel meet to 0
    # Fourth, We found the result by proportioning the contour of the discontinuous channel to the original contour of the channel.
    def calculate_discontinuity(M3_pixels, M3_channel_pixels):
        if np.array_equal(M3_pixels, []):
            discontinuity = -1
        else:
            M3_pixels[M3_pixels > 0] = 255
            M3_channel_pixels[M3_channel_pixels > 0] = 255

            contours, _ = cv2.findContours(M3_channel_pixels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_image = np.zeros_like(M3_channel_pixels)
            cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 2)

            reference = np.where(contour_image > 0, contour_image, 0)
            observed = np.where(contour_image == M3_pixels, 0, contour_image)

            reference_count = np.count_nonzero(reference)
            observed_count = np.count_nonzero(observed)

            discontinuity = (1 - (observed_count / reference_count))

        return discontinuity

    right_discontinuity = calculate_discontinuity(right_M3_pixels, right_M3_channel_pixels)
    left_discontinuity = calculate_discontinuity(left_M3_pixels, left_M3_channel_pixels)

    return right_discontinuity, left_discontinuity
