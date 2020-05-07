import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys


def contrast_stretching(pixel, min_value, max_value, limit_inf=0, limit_sup=255):
    return int((pixel-min_value)*((limit_sup-limit_inf)/(max_value-min_value)) + limit_inf)


def get_ranges(list_colors):
    least_value = 0
    most_value = 0
    global_state = True
    for b in range(len(list_colors)):
        if global_state:
            if list_colors[b] != 0:
                least_value = b
                global_state = False
        else:
            if list_colors[b] != 0:
                most_value = b
    return least_value, most_value


def show_histogram(histogram, show):
    if show:
        plt.show()
    return get_ranges(histogram[0])


def get_histogram(img, show):
    f = plt.hist(img.ravel(), 256, [0, 256])
    return show_histogram(f, show)


def get_ranges_limits(values, l1, l2):
    print(values)
    amount = values[1] - values[0]
    individual_value = amount / 100
    return int(values[0] + l1*individual_value), int(values[1] - (100-l2) * individual_value)


def solve(img, limit1=0, limit2=100):
    first, second = get_ranges_limits(get_histogram(img, True), limit1, limit2)
    print(first)
    print(second)
    cols, rows = img.shape
    for i in range(rows):
        for j in range(cols):
            if first <= img[i][j] <= second:
                img[i][j] = contrast_stretching(img[i][j], first, second)
    cv.imshow("Transform Image with limits", img)


def name_to_save_image(a):
    if a == 1:
        return "Image_converted"
    elif a == 2:
        return "Image_with_outlier"
    elif a == 3:
        return "Image_converted_part_3"
    else:
        return "Image_with_limits"


def add_outliers(img):
    cols, rows = img.shape
    range1 = int(cols / 10)
    range2 = int(rows / 10)
    for i in range(range2):
        for j in range(range1):
            img[i][j] = 0
    cv.imshow("Imagen transformada", img)


original = cv.imread("Image_with_outlier.png")

state = 5

if original is None:
    sys.exit("No se puede leer la imagen")

image = cv.cvtColor(original, cv.COLOR_BGR2GRAY)


cv.imshow("Main Image", image)
if state == 5:
    j = get_histogram(image, True)
else:
    if state == 2:
        add_outliers(image)

    if state == 4:
        solve(image, 42)
    else:
        solve(image)

k = cv.waitKey(0)
if k == ord("s"):
    name = name_to_save_image(state) + ".png"
    cv.imwrite(name, image)
