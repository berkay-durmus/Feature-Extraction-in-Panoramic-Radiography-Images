import matplotlib.pyplot as plt
import numpy as np
import cv2


def getPixels_of_image(input_image, right_48_coordinates, right_M3_channel_coordinates, left_38_coordinates,
                       left_M3_channel_coordinates, image_path):
    def fill(image_path, input_image_coord):
        """
        This part is the same function with matlab fill function
        """
        new_orgin = plt.imread(image_path, 0)
        if len(new_orgin.shape) == 3:
            new_orgin = cv2.cvtColor(new_orgin, cv2.COLOR_BGR2GRAY)

        new2 = new_orgin.copy()
        new_orgin = new2.copy()
        polygon_points = np.array([input_image_coord])
        cv2.fillPoly(new_orgin, [polygon_points], 255)
        mask = np.zeros((new_orgin.shape[0], new_orgin.shape[1]), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon_points], 255)
        new_orgin[np.copy(mask) == 0] = 0
        temp = np.bitwise_and(new_orgin, new2)

        return temp

    # This part is stored pixels of the teeth and canals
    # First, The border is drawn in the for loops as bellow
    # Second, The inside of border is painted by fill() function
    # Third, In order to getting original pixels , The intersection is taken between painted image and original image
    # Finally, The results are returned

    right_48_matrix = np.zeros(input_image.shape)
    right_m3_channel = np.zeros(input_image.shape)
    left_38_matrix = np.zeros(input_image.shape)
    left_m3_channel = np.zeros(input_image.shape)

    for temp1 in range(len(right_48_coordinates)):
        if temp1 != len(right_48_coordinates) - 1:
            cv2.line(input_image, right_48_coordinates[temp1], right_48_coordinates[temp1 + 1], 255, 2)
            cv2.line(right_48_matrix, right_48_coordinates[temp1], right_48_coordinates[temp1 + 1], 255, 2)
        else:
            cv2.line(input_image, right_48_coordinates[temp1], right_48_coordinates[0], 255, 2)
            cv2.line(right_48_matrix, right_48_coordinates[temp1], right_48_coordinates[0], 255, 2)

    for temp2 in range(len(right_M3_channel_coordinates)):
        if temp2 != len(right_M3_channel_coordinates) - 1:
            cv2.line(input_image, right_M3_channel_coordinates[temp2], right_M3_channel_coordinates[temp2 + 1], 255, 2)
            cv2.line(right_m3_channel, right_M3_channel_coordinates[temp2], right_M3_channel_coordinates[temp2 + 1],
                     255, 2)
        else:
            cv2.line(input_image, right_M3_channel_coordinates[temp2], right_M3_channel_coordinates[0], 255, 2)
            cv2.line(right_m3_channel, right_M3_channel_coordinates[temp2], right_M3_channel_coordinates[0], 255, 2)

    for temp3 in range(len(left_38_coordinates)):
        if temp3 != len(left_38_coordinates) - 1:
            cv2.line(input_image, left_38_coordinates[temp3], left_38_coordinates[temp3 + 1], 255, 2)
            cv2.line(left_38_matrix, left_38_coordinates[temp3], left_38_coordinates[temp3 + 1], 255, 2)
        else:
            cv2.line(input_image, left_38_coordinates[temp3], left_38_coordinates[0], 255, 2)
            cv2.line(left_38_matrix, left_38_coordinates[temp3], left_38_coordinates[0], 255, 2)

    for temp4 in range(len(left_M3_channel_coordinates)):
        if temp4 != len(left_M3_channel_coordinates) - 1:
            cv2.line(input_image, left_M3_channel_coordinates[temp4], left_M3_channel_coordinates[temp4 + 1], 255, 2)
            cv2.line(left_m3_channel, left_M3_channel_coordinates[temp4], left_M3_channel_coordinates[temp4 + 1], 255,
                     2)
        else:
            cv2.line(input_image, left_M3_channel_coordinates[temp4], left_M3_channel_coordinates[0], 255, 2)
            cv2.line(left_m3_channel, left_M3_channel_coordinates[temp4], left_M3_channel_coordinates[0], 255, 2)

    if right_48_coordinates:
        right_48_matrix_new = fill(image_path, right_48_coordinates)
    else:
        right_48_matrix_new = []
    if right_M3_channel_coordinates:
        right_m3_channel_new = fill(image_path, right_M3_channel_coordinates)
    else:
        right_m3_channel_new = []

    if left_38_coordinates:
        left_38_matrix_new = fill(image_path, left_38_coordinates)
    else:
        left_38_matrix_new = []

    if left_M3_channel_coordinates:
        left_m3_channel_new = fill(image_path, left_M3_channel_coordinates)
    else:
        left_m3_channel_new = []

    return input_image, right_48_matrix_new, right_m3_channel_new, left_38_matrix_new, left_m3_channel_new
