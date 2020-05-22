import cv2 as cv
import numpy as np
import os
import shutil
from CG_Operadores.settings import MEDIA_ROOT


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


def solve_exponential(path, name, constant, second_constant):
    image = cv.imread(path + "/" + name)
    message_failed = "Didn't create file"
    if image is None:
        return False, message_failed
    g = np.uint8(exponential_operator(constant, second_constant, image))
    name_to_archive = "exponential_of_"+name+"_"+str(constant)+"_"+str(second_constant)+".png"
    ubication_final = MEDIA_ROOT + "/" + name_to_archive
    check_folder(ubication_final)
    cv.imwrite(ubication_final+"/"+name_to_archive, g)
    return True, name_to_archive, ubication_final


def solve_logarithm(path, name, constant):
    image = cv.imread(path + "/" + name)
    message_failed = "Didn't create file"
    if image is None:
        return False, message_failed
    g = np.uint8(logarithm_operator(constant, image))
    name_to_archive = "logarithm_of_"+name+"_"+str(constant)+".png"
    ubication_final = MEDIA_ROOT + "/" + name_to_archive
    check_folder(ubication_final)
    cv.imwrite(ubication_final + "/" + name_to_archive, g)
    return True, name_to_archive, ubication_final


def solve_raise_power(path, name, constant, second_constant):
    image = cv.imread(path + "/" + name)
    message_failed = "Didn't create file"
    if image is None:
        return False, message_failed
    g = np.uint8(raise_power_operator(constant, second_constant, image))
    name_to_archive = "raise_power_of_"+name+"_"+str(constant)+"_"+str(second_constant)+".png"
    ubication_final = MEDIA_ROOT + "/" + name_to_archive
    check_folder(ubication_final)
    cv.imwrite(ubication_final + "/" + name_to_archive, g)
    return True, name_to_archive, ubication_final


def solve_square_root(path, name, constant):
    image = cv.imread(path + "/" + name)
    message_failed = "Didn't create file"
    if image is None:
        return False, message_failed
    g = np.uint8(square_root_operator(constant, image))
    name_to_archive = "square_root_of_"+name+"_"+str(constant)+".png"
    ubication_final = MEDIA_ROOT + "/" + name_to_archive
    check_folder(ubication_final)
    cv.imwrite(ubication_final + "/" + name_to_archive, g)
    return True, name_to_archive, ubication_final


def solve_thresholding(path, name, constant, constant1):
    original = cv.imread(path + "/" + name)
    message_failed = "Didn't create file"
    if original is None:
        return False, message_failed

    image = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
    g = np.uint8(thresholding_operator(image, constant, constant1))
    name_to_archive = "thresholding_of_"+name+"_"+str(constant)+"_"+str(constant1)+".png"
    ubication_final = MEDIA_ROOT + "/" + name_to_archive
    check_folder(ubication_final)
    cv.imwrite(ubication_final + "/" + name_to_archive, g)
    return True, name_to_archive, ubication_final


def get_original_file_extra(path, name):
    for obj in os.listdir(path):
        if os.path.isfile(path + "/" + obj):
            if os.path.splitext(obj)[0] == name:
                return obj


def check_folder(ubication_final):
    if os.path.isdir(ubication_final):
        shutil.rmtree(ubication_final)
    os.mkdir(ubication_final)