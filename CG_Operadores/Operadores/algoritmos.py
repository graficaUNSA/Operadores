import numpy as np


def histogram_equalization(img, data):
    return np.where(img, data[img], 0)


def contrast_stretching_operator(image_con, values, limit_inf=0, limit_sup=255):
    for i in range(3):
        diference = values[i][1]-values[i][0]
        image_con[:, :, i] = (image_con[:, :, i] - values[i][0]) * ((limit_sup-limit_inf)/diference) + limit_inf
    return image_con


def raise_power_operator(constant, second_constant, data_pixel):
    return constant * np.power(data_pixel, second_constant)


def exponential_operator(constant, second_constant, data_pixel):
    return constant * (np.power(second_constant, data_pixel) - 1)


def logarithm_operator(constant, data_pixel):
    return constant * np.log10(1 + data_pixel)


def square_root_operator(constant, data_pixel):
    return constant * np.power(data_pixel, 0.5)


def thresholding_operator(img, r1, r2):
    return np.where((r1 <= img) & (img <= r2), 255, 0)


def add_pixel(img1, img2):
    return img1 + img2


def difference_pixel(img1, img2):
    return np.abs(img1 - img2)


def dot_images(img1, img2):
    return img1 * img2


def division_image(img1, img2):
    return img1/img2


def blinding_image(img1, img2, var_x):
    return var_x*img1 + (1-var_x)*img2


def and_operator(img1, img2):
    return np.bitwise_and(img1, img2)


def or_operator(img1, img2):
    return np.bitwise_or(img1, img2)


def xor_operator(img1, img2):
    return np.bitwise_xor(img1, img2)


def not_operator(img1):
    return np.bitwise_not(img1)