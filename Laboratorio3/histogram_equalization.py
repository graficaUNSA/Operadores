import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt
import sys


def histogram_equalization(img, data):
    cols, rows = img.shape
    for i in range(cols):
        for j in range(rows):
            img[i][j] = data[img[i][j]]


def get_probabilities(values, total):
    p = [0] * len(values)
    for i in range(len(values)):
        p[i] = values[i]/total
    return p


def get_histogram(img):
    f = plt.hist(img.ravel(), 256, [0, 256])
    return f[0]


def get_new_intensities(img, show, amount_bits=8):
    values = get_histogram(img)
    if show:
        plt.show()
    cols, rows = img.shape
    length_pixels = math.pow(2, amount_bits)
    probabilities = get_probabilities(values, cols*rows)
    intensities = [0]*len(probabilities)
    for i in range(len(probabilities)):
        accumulate = probabilities[i]
        for j in range(0, i):
            accumulate += probabilities[j]
        intensities[i] = math.floor((length_pixels - 1) * accumulate)

    return intensities


def solve(img, show, sub_image=False, p_start1=None, p_end1=None):
    data = []
    if sub_image:
        cols = p_end1[0] - p_start1[0]
        rows = p_end1[1] - p_start1[1]
        blank_image = np.zeros((cols, rows), np.uint8)
        for i in range(cols):
            for j in range(rows):
                blank_image[i][j] = img[p_start1[0]+i][p_start1[1]+j]

        #cv.imshow("Subimage getting", blank_image)
        data = get_new_intensities(blank_image, show)
    else:
        data = get_new_intensities(img, show)
    histogram_equalization(img, data)
    cv.imshow("Image changed", img)


def name_to_save_image(a):
    if a == 1:
        return "Image_converted"
    elif a == 2:
        return "Image_converted_2"
    elif a == 3:
        return "Image_chapel_1"
    else:
        return "Image_chapel_2"


original = cv.imread("E2_hist10_1.jpg")

state = 4
show_last = False

if original is None:
    sys.exit("Can't read the image")

image = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
cv.imshow("Main Image", image)
if show_last:
    g = get_histogram(image)
    plt.show()
else:
    print(image.shape)
    # p_start = (202, 298)
    p_start = (298, 186)
    # p_end = (234, 364)
    p_end = (364, 255)
    solve(image, True, True, p_start, p_end)

k = cv.waitKey(0)
if k == ord("s"):
    name = name_to_save_image(state) + ".png"
    cv.imwrite(name, image)
