import cv2 as cv
import numpy as np
import os


def raise_power_operator(constant, second_constant, data_pixel):
    return constant * np.power(data_pixel, second_constant)


def exponential_operator(constant, second_constant, data_pixel):
    return constant * (np.power(second_constant, data_pixel) - 1)


def logarithm_operator(constant, data_pixel):
    return constant * np.log10(1 + data_pixel)


def get_original_file_extra(path):
    return [obj for obj in os.listdir(path) if os.path.isfile(path + "/" + obj)][0]


def solve(path, name, check, constant=0, second_constant=0):
    image = cv.imread(path + "/" + name)
    if check == 1:
        g = np.uint8(exponential_operator(constant, second_constant, np.copy(image)))
        cv.imwrite(path+"/exponential.png", g)
    else:
        print("nada")

