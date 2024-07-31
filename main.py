import cv2
import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter, re, os, time
import sys

from obscuration import obscuration
from createMask import getPixels_of_image
from deflectionRoots import deflectionRoots
from deflectionMandibularCanal import deflection_mandibular_canal
from narrowingRoots import narrowing_roots
from narrowingMandibularCanal import narrowing_mandibular_canal
from discontinuityCanal import discontinuity_canal
from extractCoordinate import get_Coordinates

"""
-----------------------------------------------------------------------------------------------------
BERKAY AHMET DURMUŞ

NOTE : JUST RUN THE CODE , IF YOU TRY ANOTHER IMAGE FOR GETTING RESULT  , GO MAIN FUNCTION AND COPY FOLDER PATH 
-------------------------------------------------------------------------------------------------------
"""


def main():
    t1 = time.time()
    book = xlsxwriter.Workbook('FeatureList.xlsx')
    page = book.add_worksheet()

    page.write(0, 0, 'ImageNumber')
    page.write(0, 1, 'Tooth')
    page.write(0, 2, 'Obscuration')
    page.write(0, 3, 'Deflection')
    page.write(0, 4, 'NarrowingRoots')
    page.write(0, 5, 'NarrowingCanal')
    page.write(0, 6, 'DiscontinuityCanal')
    page.write(0, 7, 'CanalDeviation')

    Image_Debug = False  # If you displayed the images, Changing False by True

    folder_path = input("Please Enter Image File Path : ")
    folder_path = folder_path[1:-1]
    xml_files = sorted([file for file in os.listdir(folder_path) if file.endswith(".xml")],
                       key=lambda x: int(re.search(r'\d+', x).group()))
    image_files = sorted([file for file in os.listdir(folder_path) if file.endswith(".jpg")],
                         key=lambda x: int(re.search(r'\d+', x).group()))

    row = 1
    for image_file, xml_file in zip(image_files, xml_files):
        try:
            img_no = image_file[:-4]
            image_path = os.path.join(folder_path, image_file)
            xml_path = os.path.join(folder_path, xml_file)
            right_M3_channel_coordinates, right_48_coordinates, left_M3_channel_coordinates, left_38_coordinates = get_Coordinates(
                xml_path)
            print(f"Number: {img_no} .. Feature extraction is running please wait ...")
            img = plt.imread(image_path, format='gray')
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            I_drawing_bordered, right_48_pixels, right_M3_channel_pixels, left_38_pixels, left_M3_channel_pixels = getPixels_of_image(
                img, right_48_coordinates,
                right_M3_channel_coordinates,
                left_38_coordinates,
                left_M3_channel_coordinates, image_path)
            if Image_Debug:
                if not np.array_equal(I_drawing_bordered, []):
                    plt.imshow(I_drawing_bordered, cmap="gray")
                plt.figure()
                if not np.array_equal(right_48_pixels, []):
                    plt.imshow(right_48_pixels, cmap="gray")
                plt.figure()
                if not np.array_equal(right_M3_channel_pixels, []):
                    plt.imshow(right_M3_channel_pixels, cmap="gray")
                plt.figure()
                if not np.array_equal(left_38_pixels, []):
                    plt.imshow(left_38_pixels, cmap="gray")
                plt.figure()
                if not np.array_equal(left_M3_channel_pixels, []):
                    plt.imshow(left_M3_channel_pixels, cmap="gray")
                plt.show()

            obs_48, obs_38 = obscuration(right_48_pixels, right_M3_channel_pixels, left_38_pixels,
                                         left_M3_channel_pixels)
            deflect_48_t, deflect_38_t = deflectionRoots(right_48_pixels, left_38_pixels)
            result_48_narrowing_roots, result_38__narrowing_roots = narrowing_roots(right_48_pixels, left_38_pixels)
            nr_C_48, nr_C_38 = narrowing_mandibular_canal(right_M3_channel_pixels, left_M3_channel_pixels)
            dc_canal_48, dc_canal_38 = discontinuity_canal(right_48_pixels, right_M3_channel_pixels, left_38_pixels,
                                                           left_M3_channel_pixels)
            C_deviation_48, C_deviation_38 = deflection_mandibular_canal(right_M3_channel_pixels,
                                                                         left_M3_channel_pixels)

            page.write(row, 0, img_no)
            page.write(row, 1, '48')
            page.write(row, 2, obs_48)
            page.write(row, 3, deflect_48_t)
            page.write(row, 4, result_48_narrowing_roots)
            page.write(row, 5, nr_C_48)
            page.write(row, 6, dc_canal_48)
            page.write(row, 7, C_deviation_48)
            row = row + 1
            page.write(row, 0, img_no)
            page.write(row, 1, '38')
            page.write(row, 2, obs_38)
            page.write(row, 3, deflect_38_t)
            page.write(row, 4, result_38__narrowing_roots)
            page.write(row, 5, nr_C_38)
            page.write(row, 6, dc_canal_38)
            page.write(row, 7, C_deviation_38)
            row = row + 1
        except Exception as e:
            print(f"Any error was occurred: {e}")

    book.close()
    t2 = time.time() - t1
    print("The Feature Extraction is completed . Thank you for your patience (:")
    print("The programme is terminating")

    print(f"The program run in {t2} seconds")

    sys.stdout.write("Press any key for quit..")
    sys.stdout.flush()
    sys.stdin.read(1)


if __name__ == "__main__":
    main()
