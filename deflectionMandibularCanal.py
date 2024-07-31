import cv2
import statistics
import numpy as np
from skimage import morphology


def deflection_mandibular_canal(right_m3_channel_new, left_m3_channel_new):
    # This part find deflection mandibular canal
    # First, The skeleton of Canal is found by morphology.skeletonize()
    # Second, The largest contour of the skeleton is found by cv2.findContours()
    # Third, The Contour points are converted into 1-D vector(list) by largest_contour.squeeze().tolist()
    # Fourth, To Finding deflection mandibular canal , The derivation process is made
    # In Short, The index change of the pixels on the skeletonized image gives us the derivatives and we find deviation thanks to the derivatives
    def CanalDeviation(image):

        kernel = np.ones((3, 3))
        img = image.copy()
        img = cv2.erode(img, kernel=kernel, iterations=5)

        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        binary = np.where(img > 0, 1, img)
        skeleton = morphology.skeletonize(binary, method='lee')
        skeleton = np.where(skeleton == 1, 255, skeleton)

        contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_contour = max(contours, key=cv2.contourArea)

        points = largest_contour.squeeze().tolist()

        del points[:50]
        del points[len(points) - 100:]

        sayac = 0
        derivatives = []
        for i in range(len(points) - 1):
            if len(points) - 1 < 10:
                sayac = 1

            x1, y1 = points[i + sayac]
            sayac += 10
            if i + sayac >= len(points) - 1:
                break
            x2, y2 = points[i + sayac]
            dx = x2 - x1
            dy = y2 - y1

            if dx == 0:
                k = 0
            else:
                derivatives.append(abs(dy / dx))

        result = statistics.stdev(derivatives)
        return result

    if np.array_equal(right_m3_channel_new, []):
        C_deviation_48 = -1
    else:
        C_deviation_48 = CanalDeviation(right_m3_channel_new)
    if np.array_equal(left_m3_channel_new, []):
        C_deviation_38 = -1
    else:
        C_deviation_38 = CanalDeviation(left_m3_channel_new)
    return C_deviation_48, C_deviation_38
