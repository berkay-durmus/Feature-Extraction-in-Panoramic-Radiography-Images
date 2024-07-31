import scipy.ndimage as ndi
import numpy as np
import cv2

"""
Functions that implement some of the same functionality found in Matlab's bwmorph.
`_neighbors_conv,branches` - was taken and adapted from https://gist.github.com/bmabey
"""


def _neighbors_conv(image):
    """
    Counts the neighbor pixels for each pixel of an image:
            x = [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]
            ]
            _neighbors(x)
            [
                [0, 3, 0],
                [3, 4, 3],
                [0, 3, 0]
            ]
    :type image: numpy.ndarray
    :param image: A two-or-three dimensional image
    :return: neighbor pixels for each pixel of an image
    """
    image = image.astype(np.int_)
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neighborhood_count = ndi.convolve(image, k, mode='constant', cval=1)
    neighborhood_count[~image.astype(np.bool_)] = 0
    return neighborhood_count


def branches(image):
    """
    Returns the nodes in between edges
    Parameters
    ----------
    image : binary (M, N) ndarray
    Returns
    -------
    out : ndarray of bools
        image.
    """
    return _neighbors_conv(image) > 2


def reduce_area(img):
    """
    -> This function returns just related object area
    -> To find corner of the image , The Harris Corner method was used
    """

    corner_detector = cv2.cornerHarris(img, blockSize=2, ksize=3, k=0.04)

    corner_detector = cv2.dilate(corner_detector, None)

    threshold = 0.01 * corner_detector.max()
    corner_points = np.where(corner_detector > threshold)

    min_x = np.min(corner_points[1])
    min_y = np.min(corner_points[0])
    max_x = np.max(corner_points[1])
    max_y = np.max(corner_points[0])

    x = min_x
    y = min_y
    width = max_x - min_x
    height = max_y - min_y

    tooth = img[y - 50:y + height + 50, x - 50:x + width + 50]
    return tooth

def rotationImage(cv_image, angle):
    """
    -> This function returns rotated image by required angle
    """
    height, width = cv_image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(cv_image, rotation_matrix, (width, height))

    return rotated_image


def calculateDerivation(img):
    """
    -> This function returns average slope of the object
    -> First, The white pixels coordinate are found by np.where() method
    -> X and Y coordiantes are seperated
    -> The slope is calculated once a ten pixels
    -> All pixels value is stored in derivates list
    -> The average slope is found
    """
    result = np.where(img == 255)
    y = result[1]
    x = result[0]

    skeleton_coor_matrix = []
    for temp, i in enumerate(range(len(x))):
        skeleton_coor_matrix.append([x[i], y[i]])

    derivatives = []
    points = skeleton_coor_matrix
    sayac = 0
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

    if len(derivatives) != 0:
        result = sum(derivatives) / len(derivatives)
    else:
        result = 0

    return result


def seperation_of_Branches(temp_image):
    """
    -> The function returns branch number and each branches matrix
    -> The branches points are found by branches method
    -> The branches are labeled by opencv
    -> Every branches are got seperately
    -> To making process on the branches , the branches type is converted uint8 format
    """
    branch_points = branches(temp_image)

    num_branches, labeled_image = cv2.connectedComponents(branch_points.astype(np.uint8) * 255)

    branch_image_1 = np.zeros_like(temp_image)
    branch_image_1[labeled_image == 1] = 255
    branch_image_2 = np.zeros_like(temp_image)
    branch_image_2[labeled_image == 2] = 255

    branch_image_1 = branch_image_1.astype(np.uint8) * 255
    branch_image_2 = branch_image_2.astype(np.uint8) * 255

    return num_branches, branch_image_1, branch_image_2


def getOrientationObeject(img):
    """
    -> The function returns angle and sizing information of the image
    -> The countour of the image is found
    -> According to contour information, a rectangle is drawn around the contour
    -> The rectangle box points are found and the angle is got
    """

    # Convert image to grayscale
    gray = img.copy()
    # Convert image to binary
    _, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Find all the contours in the thresholded image
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in contours:

        area = cv2.contourArea(c)

        # This part eliminate the too small or big contour (This is just essential for accuracy result)
        if area < 3700 or 100000 < area:
            continue

        # (center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(c)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        # Retrieve the key parameters of the rotated bounding box
        center = (int(rect[0][0]), int(rect[0][1]))
        width = int(rect[1][0])
        height = int(rect[1][1])
        angle = int(rect[2])

        if width < height:
            angle = 90 - angle
        else:
            angle = -angle

    return angle, center, width, height
