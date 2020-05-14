import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt
import sys


def logarithm_operator(constant, data_pixel):
    return constant * math.log(1 + data_pixel, 10)


def square_root_operator(constant, data_pixel):
    return constant * math.sqrt(data_pixel)


def solve(img, constant):
    cols, rows = img.shape
    for i in range(cols):
        for j in range(rows):
            img[i][j] = logarithm_operator(constant, img[i][j])
    cv.imshow("Image changed with constant " + str(constant), img)


def name_to_save_image(a, constant):
    return "Image_result_" + str(a) + "_" + str(constant)


original = cv.imread("log_1.jpg")

state = 1
show_last = False

if original is None:
    sys.exit("Can't read the image")

image = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
cv.imshow("Main Image", image)
value = 200
solve(image, value)

k = cv.waitKey(0)
if k == ord("s"):
    name = name_to_save_image(state, value) + ".png"
    cv.imwrite(name, image)
