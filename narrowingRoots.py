import numpy as np
import cv2
from skimage import morphology
import matplotlib.pyplot as plt
from utils import reduce_area, getOrientationObeject, seperation_of_Branches, rotationImage

def narrowing_roots(right_M3_pixels, left_M3_pixels):
    """
    -> This funtion returns information of the narrowing (1 0 -1)
    -> -1 means there is no information , 1 means there is narrowing on the root , 0 means there is no narrowing on the roots
    -> The image rotated perpendicularly
    -> The skeleton of the image is found
    -> The branches of the skeleton is found
    -> If the Polygone is single , then function returns -1 because The information cannot extracting from single polygone
    -> X and Y coordinates of the branche are found for getting orientation
    -> If the branches are oriented on the y axes , max and min y coordiante is found , then divided by 2
    -> The branche is cut mid-point and the area is calculated each of cutting region, and they are compared
    -> If there is narrowing on the root , the area of the root should be less 10% than body of the branche
    -> The same procedure valid x-axes
    """

    def Right_48_tooth(right_M3_pixels):
        if np.array_equal(right_M3_pixels, []):
            return -1
        else:
            reduction = reduce_area(right_M3_pixels)
            kernel = np.ones((3, 3))
            angle, center, height, width = getOrientationObeject(reduction)
            img = cv2.erode(reduction, kernel=kernel, iterations=4)

            if angle < 0:
                rotated_image = rotationImage(img, 90 + abs(angle))
            else:
                rotated_image = rotationImage(img, 90 - angle)
            plt.figure()
            plt.imshow(rotated_image, cmap='gray')

            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            binary = np.where(img > 0, 1, img)
            skeleton = morphology.skeletonize(binary, method='lee')
            skeleton = skeleton.astype(np.uint8) * 255
            skeleton[:(skeleton.shape[0] // 2) + 30, :] = 0
            temp_skeleton = skeleton.copy()

            num_branches, branch_image_1, branch_image_2 = seperation_of_Branches(temp_skeleton)
            branch_image_1 = branch_image_1.astype(np.uint8) * 255
            branch_image_2 = branch_image_2.astype(np.uint8) * 255
            coor_points_initial_br1 = []

            if num_branches == 2:
                return -1
            else:
                try:
                    coor_points_initial_br1 = [i for i in range(branch_image_1.shape[0]) if
                                               any(branch_image_1[i, j] > 0 for j in range(branch_image_1.shape[1]))]

                    rotated_image[:int((np.median(coor_points_initial_br1))), :] = 0
                    rotated_image = rotated_image.astype(np.uint8) * 255

                    num_branches_1, branch_image_1_1, branch_image_2_1 = seperation_of_Branches(rotated_image)

                    coor_points_initial_1_x = [i for i in range(branch_image_1_1.shape[0]) if any(
                        branch_image_1_1[i, j] > 0 for j in range(branch_image_1_1.shape[1]))]
                    coor_points_initial_2_x = [i for i in range(branch_image_2_1.shape[0]) if any(
                        branch_image_2_1[i, j] > 0 for j in range(branch_image_2_1.shape[1]))]
                    coor_points_initial_1_y = [j for j in range(branch_image_1_1.shape[1]) if any(
                        branch_image_1_1[i, j] > 0 for i in range(branch_image_1_1.shape[0]))]
                    coor_points_initial_2_y = [j for j in range(branch_image_2_1.shape[1]) if any(
                        branch_image_2_1[i, j] > 0 for i in range(branch_image_2_1.shape[0]))]

                    b1 = branch_image_1_1.copy()
                    b2 = branch_image_1_1.copy()

                    b3 = branch_image_2_1.copy()
                    b4 = branch_image_2_1.copy()

                    if len(coor_points_initial_1_x) > len(coor_points_initial_1_y):
                        coor_points_initial_1 = coor_points_initial_1_x
                        b1[int((np.max(coor_points_initial_1) + np.min(coor_points_initial_1)) // 2):, :] = 0
                        b2[:int((np.max(coor_points_initial_1) + np.min(coor_points_initial_1)) // 2), :] = 0
                    else:
                        coor_points_initial_1 = coor_points_initial_1_y
                        mid_point = int(np.median(coor_points_initial_1))
                        if np.max(coor_points_initial_1) > mid_point * 1.1:
                            b1[:, mid_point:] = 0
                            b2[:, :mid_point] = 0
                        else:
                            b2[:, mid_point:] = 0
                            b1[:, :mid_point] = 0

                    if len(coor_points_initial_2_x) > len(coor_points_initial_2_y):
                        coor_points_initial_2 = coor_points_initial_2_x
                        b3[int((np.max(coor_points_initial_2) + np.min(coor_points_initial_2)) // 2):, :] = 0
                        b4[:int((np.max(coor_points_initial_2) + np.min(coor_points_initial_2)) // 2), :] = 0
                    else:
                        coor_points_initial_2 = coor_points_initial_2_y
                        mid_point = int(np.median(coor_points_initial_2))
                        if np.max(coor_points_initial_2) > mid_point * 1.1:
                            b3[:, mid_point:] = 0
                            b4[:, :mid_point] = 0
                        else:
                            b4[:, mid_point:] = 0
                            b3[:, :mid_point] = 0

                    sayac1 = 0
                    sayac2 = 0
                    sayac3 = 0
                    sayac4 = 0

                    sayac1 = sum(1 for i in range(b1.shape[0]) for j in range(b1.shape[1]) if b1[i, j] > 0)
                    sayac2 = sum(1 for i in range(b2.shape[0]) for j in range(b2.shape[1]) if b2[i, j] > 0)
                    sayac3 = sum(1 for i in range(b3.shape[0]) for j in range(b3.shape[1]) if b3[i, j] > 0)
                    sayac4 = sum(1 for i in range(b4.shape[0]) for j in range(b4.shape[1]) if b4[i, j] > 0)

                    if sayac2 < sayac1:
                        if sayac2 < sayac1 * 0.8:
                            # print("first 48 root {}".format(1))
                            result_first = 1
                        else:
                            # print("first 48 root {}".format(0))
                            result_first = 0

                    else:
                        if sayac1 < sayac2 * 0.8:
                            # print("first 48 root {}".format(1))
                            result_first = 1
                        else:
                            # print("first 48 root {}".format(0))
                            result_first = 0

                    if sayac4 < sayac3:
                        if sayac4 < sayac3 * 0.8:
                            # print("second 48 root {}".format(1))
                            result_second = 1
                        else:
                            # print("second 48 root {}".format(0))
                            result_second = 0
                    else:
                        if sayac3 < sayac4 * 0.8:
                            # print("second 48 root {}".format(1))
                            result_second = 1
                        else:
                            # print("second 48 root {}".format(0))
                            result_second = 0
                    if result_first > result_second:
                        return result_first
                    else:
                        return result_second
                except:
                    return -1

    def Left_38_tooth(left_M3_pixels):
        if np.array_equal(left_M3_pixels, []):
            return -1

        else:
            reduction = reduce_area(left_M3_pixels)
            kernel = np.ones((3, 3))
            angle, center, height, width = getOrientationObeject(reduction)
            img = cv2.erode(reduction, kernel=kernel, iterations=4)

            if angle < 0:
                rotated_image = rotationImage(img, 270 + abs(angle))
            else:
                rotated_image = rotationImage(img, 270 - angle)

            img = cv2.morphologyEx(rotated_image, cv2.MORPH_OPEN, kernel)
            binary = np.where(img > 0, 1, img)

            skeleton = morphology.skeletonize(binary, method='lee')
            skeleton = skeleton.astype(np.uint8) * 255
            skeleton[:(skeleton.shape[0] // 2) + 30, :] = 0
            temp_skeleton = skeleton.copy()

            num_branches, branch_image_1, branch_image_2 = seperation_of_Branches(temp_skeleton)

            if num_branches == 2:
                return -1
            else:
                try:
                    coor_points_initial_br1 = [i for i in range(branch_image_1.shape[0]) if
                                               any(branch_image_1[i, j] > 0 for j in range(branch_image_1.shape[1]))]
                    rotated_image[:int((np.median(coor_points_initial_br1)) - 5), :] = 0

                    num_branches_1, branch_image_1_1, branch_image_2_1 = seperation_of_Branches(rotated_image)

                    coor_points_initial_1_x = [i for i in range(branch_image_1_1.shape[0]) if any(
                        branch_image_1_1[i, j] > 0 for j in range(branch_image_1_1.shape[1]))]
                    coor_points_initial_2_x = [i for i in range(branch_image_2_1.shape[0]) if any(
                        branch_image_2_1[i, j] > 0 for j in range(branch_image_2_1.shape[1]))]
                    coor_points_initial_1_y = [j for j in range(branch_image_1_1.shape[1]) if any(
                        branch_image_1_1[i, j] > 0 for i in range(branch_image_1_1.shape[0]))]
                    coor_points_initial_2_y = [j for j in range(branch_image_2_1.shape[1]) if any(
                        branch_image_2_1[i, j] > 0 for i in range(branch_image_2_1.shape[0]))]

                    b1 = branch_image_1_1.copy()
                    b2 = branch_image_1_1.copy()

                    b3 = branch_image_2_1.copy()
                    b4 = branch_image_2_1.copy()

                    if len(coor_points_initial_1_x) > len(coor_points_initial_1_y):
                        coor_points_initial_1 = coor_points_initial_1_x
                        b1[int((np.max(coor_points_initial_1) + np.min(coor_points_initial_1)) // 2):, :] = 0
                        b2[:int((np.max(coor_points_initial_1) + np.min(coor_points_initial_1)) // 2), :] = 0
                    else:
                        coor_points_initial_1 = coor_points_initial_1_y
                        mid_point = int(np.median(coor_points_initial_1))
                        if np.max(coor_points_initial_1) > mid_point * 1.1:
                            b1[:, mid_point:] = 0
                            b2[:, :mid_point] = 0
                        else:
                            b2[:, mid_point:] = 0
                            b1[:, :mid_point] = 0

                    if len(coor_points_initial_2_x) > len(coor_points_initial_2_y):
                        coor_points_initial_2 = coor_points_initial_2_x
                        b3[int((np.max(coor_points_initial_2) + np.min(coor_points_initial_2)) // 2):, :] = 0
                        b4[:int((np.max(coor_points_initial_2) + np.min(coor_points_initial_2)) // 2), :] = 0
                    else:
                        coor_points_initial_2 = coor_points_initial_2_y
                        mid_point = int(np.median(coor_points_initial_2))
                        if np.max(coor_points_initial_2) > mid_point * 1.1:
                            b3[:, mid_point:] = 0
                            b4[:, :mid_point] = 0
                        else:
                            b4[:, mid_point:] = 0
                            b3[:, :mid_point] = 0

                    sayac1 = sum(1 for i in range(b1.shape[0]) for j in range(b1.shape[1]) if b1[i, j] > 0)
                    sayac2 = sum(1 for i in range(b2.shape[0]) for j in range(b2.shape[1]) if b2[i, j] > 0)
                    sayac3 = sum(1 for i in range(b3.shape[0]) for j in range(b3.shape[1]) if b3[i, j] > 0)
                    sayac4 = sum(1 for i in range(b4.shape[0]) for j in range(b4.shape[1]) if b4[i, j] > 0)
                    if sayac2 < sayac1:
                        if sayac2 < sayac1 * 0.8:
                            # print("first 38 root {}".format(1))
                            result_first = 1
                        else:
                            # print("first 38 root {}".format(0))
                            result_first = 0

                    else:
                        if sayac1 < sayac2 * 0.8:
                            # print("first 38 root {}".format(1))
                            result_first = 1
                        else:
                            # print("first 38 root {}".format(0))
                            result_first = 0

                    if sayac4 < sayac3:
                        if sayac4 < sayac3 * 0.8:
                            # print("second 38 root {}".format(1))
                            result_second = 1
                        else:
                            # print("second 38 root {}".format(0))
                            result_second = 0
                    else:
                        if sayac3 < sayac4 * 0.8:
                            # print("second 38 root {}".format(1))
                            result_second = 1
                        else:
                            # print("second 38 root {}".format(0))
                            result_second = 0
                    if result_first > result_second:
                        return result_first
                    else:
                        return result_second
                except:
                    return 0

    result_38 = Left_38_tooth(left_M3_pixels)
    result_48 = Right_48_tooth(right_M3_pixels)

    return result_48, result_38