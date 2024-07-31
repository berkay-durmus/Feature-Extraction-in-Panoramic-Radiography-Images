import numpy as np
import cv2
from skimage import morphology
from utils import reduce_area, getOrientationObeject, calculateDerivation, seperation_of_Branches, rotationImage


def deflectionRoots(right_M3_pixels, left_M3_pixels):
    # This part is found deflection of Roots and returns average slope of the deflection
    # The orientation of the tooth was found by getOrientationObeject function
    # The image was rotated perpendicularly
    # The Branches of roots were found by branches function
    # The branches are named
    # The slope of related branches was found by calculateDerivation
    # The maximum slope was returned between two roots

    def Right_48_tooth(right_M3_pixels):
        if np.array_equal(right_M3_pixels, []):
            return -1
        else:
            right_M3_pixels = reduce_area(right_M3_pixels)
            angle, center, height, width = getOrientationObeject(right_M3_pixels)
            label = "  Rotation Angle: " + str(angle)
            cv2.putText(right_M3_pixels, label, (center[0] - 75, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

            kernel = np.ones((3, 3))
            img = right_M3_pixels.copy()

            img = cv2.erode(img, kernel=kernel, iterations=5)
            if angle < 0:
                right_48_pixels_rot = rotationImage(img, 90 + abs(angle))
            else:
                right_48_pixels_rot = rotationImage(img, 90 - angle)

            img = cv2.morphologyEx(right_48_pixels_rot, cv2.MORPH_OPEN, kernel)
            binary = np.where(img > 0, 1, img)

            skeleton = morphology.skeletonize(binary, method='lee')
            skeleton = skeleton.astype(np.uint8) * 255
            #cv2.imshow('rotational image', right_48_pixels_rot)
            #cv2.imshow('original sekeleton', skeleton)

            skeleton[:(skeleton.shape[0] // 2) + 30, :] = 0
            temp_skeleton = skeleton.copy()

            #branch_points = branches(temp_skeleton)
            #
            #num_branches, labeled_image = cv2.connectedComponents(branch_points.astype(np.uint8))
            #
            #branch_image_1 = np.zeros_like(temp_skeleton)
            #branch_image_1[labeled_image == 1] = 255
            #
            #branch_image_2 = np.zeros_like(temp_skeleton)
            #branch_image_2[labeled_image == 2] = 255
            num_branches, branch_image_1, branch_image_2 = seperation_of_Branches(temp_skeleton)
            branch_image_1 = branch_image_1.astype(np.uint8) * 255
            branch_image_2 = branch_image_2.astype(np.uint8) * 255
            #cv2.imshow('branch1', branch_image_1)
            #cv2.imshow('branch2', branch_image_2)
            if np.max(branch_image_1) != 0:
                deriv_br1 = calculateDerivation(branch_image_1)
            else:
                deriv_br1 = -1
            if np.max(branch_image_2) != 0:
                deriv_br2 = calculateDerivation(branch_image_2)
            else:
                deriv_br2 = -1

            if deriv_br1 > deriv_br2:
                return deriv_br1
            else:
                return deriv_br2

    def Left_38_tooth(left_M3_pixels):
        if np.array_equal(left_M3_pixels, []):
            return -1
        else:
            kernel = np.ones((3, 3))
            left_M3_pixels = reduce_area(left_M3_pixels)
            angle, center, height, width = getOrientationObeject(left_M3_pixels)
            label = "  Rotation Angle: " + str(angle)
            cv2.putText(left_M3_pixels, label, (center[0] - 75, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

            img = left_M3_pixels.copy()

            img = cv2.erode(img, kernel=kernel, iterations=5)
            if angle < 0:
                left_38_pixels_rot = rotationImage(img, 270 + abs(angle))
            else:
                left_38_pixels_rot = rotationImage(img, 270 - angle)

            #cv2.imshow('rotational image', left_38_pixels_rot)
            img = cv2.morphologyEx(left_38_pixels_rot, cv2.MORPH_OPEN, kernel)
            binary = np.where(img > 0, 1, img)

            skeleton = morphology.skeletonize(binary, method='lee')
            skeleton = skeleton.astype(np.uint8) * 255

            #cv2.imshow('original sekeleton', skeleton)

            skeleton[:(skeleton.shape[0] // 2) + 30, :] = 0
            temp_skeleton = skeleton.copy()

            #branch_points = branches(temp_skeleton)
            #num_branches, labeled_image = cv2.connectedComponents(branch_points.astype(np.uint8))

            #branch_image_1 = np.zeros_like(temp_skeleton)
            #branch_image_1[labeled_image == 1] = 255

            #branch_image_2 = np.zeros_like(temp_skeleton)
            #branch_image_2[labeled_image == 2] = 255

            #cv2.imshow('branch1', branch_image_1)
            #cv2.imshow('branch2', branch_image_2)

            num_branches, branch_image_1, branch_image_2 = seperation_of_Branches(temp_skeleton)
            branch_image_1 = branch_image_1.astype(np.uint8) * 255
            branch_image_2 = branch_image_2.astype(np.uint8) * 255
            if np.max(branch_image_1) != 0:
                deriv_br1 = calculateDerivation(branch_image_1)
            else:
                deriv_br1 = -1
            if np.max(branch_image_2) != 0:
                deriv_br2 = calculateDerivation(branch_image_2)
            else:
                deriv_br2 = -1
            if deriv_br1 > deriv_br2:
                return deriv_br1
            else:
                return deriv_br2

    deflect_48 = Right_48_tooth(right_M3_pixels)
    deflect_38 = Left_38_tooth(left_M3_pixels)
    return deflect_48, deflect_38
