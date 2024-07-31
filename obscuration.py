from skimage import filters
import numpy as np


def obscuration(right_48_pixels, right_M3_channel_pixels, left_38_pixels, left_M3_channel_pixels):
    # This part includes obscuration of the root
    # Firstly, The instersections point is found between canal and tooth
    # Second The multi thresholding process is applied to the outside of the intersection
    # Third the different from 0 pixels are found and summing all of them for finding mean value of them
    # and they are stored in the result_mean_value_for_root vector
    # Fourth If there is no any intersection between the root and canal, ZeroDivisonError is occured and we know the tooth is not on the canal

    threshold_index = 2

    def calculate_obs(obs_pixels, M3_channel_pixels):
        if len(obs_pixels) == 0:
            return -1
        try:
            intersection_pixels = np.bitwise_and(obs_pixels, M3_channel_pixels)
            except_intersection = obs_pixels - intersection_pixels
            thresholds = filters.threshold_multiotsu(except_intersection, classes=5)

            valid_pixels = intersection_pixels[intersection_pixels > 0]
            if len(valid_pixels) != 0:
                result_mean_root_pixels = np.mean(valid_pixels)
            else:
                result_mean_root_pixels = thresholds[threshold_index]

            obs = (thresholds[threshold_index] - int(result_mean_root_pixels)) / (
                    thresholds[threshold_index] + int(result_mean_root_pixels))
        except:
            obs = -1

        return obs

    obs_48 = calculate_obs(right_48_pixels, right_M3_channel_pixels)
    obs_38 = calculate_obs(left_38_pixels, left_M3_channel_pixels)

    return obs_48, obs_38
