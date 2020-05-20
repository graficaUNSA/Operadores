import cv2 as cv
import numpy as np


def raise_power_operator(constant, second_constant, data_pixel):
    return constant * np.power(data_pixel, second_constant)


def exponential_operator(constant, second_constant, data_pixel):
    return constant * (np.power(second_constant, data_pixel) - 1)


def logarithm_operator(constant, data_pixel):
    return constant * np.log10(1 + data_pixel)


def solve(path, name, check, constant, second_constant):
    image = cv.imread(path + "/" + name)
    if image is None:
        print("fallo")

    if check == 1:
        g = np.uint8(exponential_operator(constant, second_constant, image))
        cv.imwrite(path+"/exponential_"+str(constant)+"_"+str(second_constant)+".png", g)


