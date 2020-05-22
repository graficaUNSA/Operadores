import cv2 as cv
import numpy as np
import os
import shutil
from CG_Operadores.settings import MEDIA_ROOT


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

#Funciones Contrast_Streching
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


def get_histogram(img):
    data = []
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv.calcHist([img], [i], None, [256], [0, 256])
        data += [get_ranges(histr)]
    return data


def get_ranges_limits(values, l1, l2):
    nvalues = []
    for i in range(len(values)):
        amount = values[i][1] - values[i][0]
        individual_value = amount / 100
        nvalues += [(int(values[i][0] + l1*individual_value), int(values[i][1] - (100-l2) * individual_value))]
    return nvalues


def solve_contrast_streching(path, name, constant, constant1):
    image = cv.imread(path + "/" + name)
    message_failed = "Didn't create file"
    if image is None:
        return False, message_failed

    val = get_ranges_limits(get_ranges(image), constant, constant1)
    g = np.uint8(contrast_stretching_operator(image, val))
    name_to_archive = "Contrast_Streching_of_"+name+"_"+str(constant)+"_"+str(constant1)+".png"
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