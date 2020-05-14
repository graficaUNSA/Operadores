import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt
import sys


def logarithm_operator(constant, data_pixel):
    return constant * (math.log(1 + data_pixel, 10))


def solve(img, constant):
    cols, rows = img.shape
    for i in range(cols):
        for j in range(rows):
            img[i][j] = logarithm_operator(constant, img[i][j])
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


original = cv.imread("log_1.jpg")

state = 4
show_last = False

if original is None:
    sys.exit("Can't read the image")

image = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
cv.imshow("Main Image", image)
solve(image, 100)

k = cv.waitKey(0)
if k == ord("s"):
    name = name_to_save_image(state) + ".png"
    cv.imwrite(name, image)
