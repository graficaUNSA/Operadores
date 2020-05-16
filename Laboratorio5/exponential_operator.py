import cv2 as cv
import numpy as np
import math
import sys


def raise_power_operator(constant, second_constant, data_pixel):
    return constant * np.power(data_pixel, second_constant)


def exponential_operator(constant, second_constant, data_pixel):
    return constant * (np.power(second_constant, data_pixel) - 1)


def logarithm_operator(constant, data_pixel):
    return constant * np.log10(1 + data_pixel)


def solve(img, constant, second_constant, check):
    g = []
    if check == 1:
        g = np.uint8(exponential_operator(constant, second_constant, img))
        cv.imshow("Image changed with exponential " + str(constant), g)
    elif check == 2:
        g = np.uint8(raise_power_operator(constant, second_constant, img))
        cv.imshow("Image changed with raise_power " + str(constant), g)
    else:
        g = np.uint8(logarithm_operator(constant, img))
        cv.imshow("Image changed with logarithm " + str(constant), g)


def name_to_save_image(a, constant):
    return "Image_result_" + str(a) + "_" + str(constant)


image = cv.imread("exp_5.jpg")

state = 1

if image is None:
    sys.exit("Can't read the image")


testing = 1
cv.imshow("Main Image", image)
value = 20
value2 = 1.01
solve(image, value, value2, testing)

k = cv.waitKey(0)
if k == ord("s"):
    name = name_to_save_image(state, value) + ".png"
    cv.imwrite(name, image)
