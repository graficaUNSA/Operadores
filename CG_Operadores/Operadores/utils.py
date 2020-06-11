import cv2 as cv
import os
import shutil
from CG_Operadores.settings import MEDIA_ROOT
from matplotlib import pyplot as plt
from .algoritmos import *
from django.core.files.storage import FileSystemStorage


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


# suma y  resta de imagenes

def rescale(img, img1, value):
    first_image = cv.resize(img, value)
    second_image = cv.resize(img1, value)
    return first_image, second_image


def get_max_values(img, img1):
    rows1 = max(img.shape[0], img1.shape[0])
    columns1 = max(img.shape[1], img1.shape[1])
    return rows1, columns1


def solve_addition(path, name, variable1):
    image = cv.imread(path + "/" + name)
    message_failed = "Didn't create file"
    image1 = variable1
    if type(variable1) != str:
        if image is None:
            return False, message_failed
    else:
        image1 = cv.imread(path + "/" + variable1)
        if image is None or image1 is None:
            return False, message_failed

        max_ranges = get_max_values(image, image1)
        image, image1 = rescale(image, image1, max_ranges)

    g = np.uint8(add_pixel(image, image1))
    name_to_archive = "Addition_of_"+name+"_"+str(variable1)+".png"
    ubication_final = MEDIA_ROOT + "/" + name_to_archive
    check_folder(ubication_final)
    cv.imwrite(ubication_final + "/" + name_to_archive, g)
    return True, name_to_archive, ubication_final


def solve_difference(path, name, variable1):
    image = cv.imread(path + "/" + name)
    message_failed = "Didn't create file"
    image1 = variable1
    if type(variable1) != str:
        if image is None:
            return False, message_failed
    else:
        image1 = cv.imread(path + "/" + variable1)
        if image is None or image1 is None:
            return False, message_failed

        max_ranges = get_max_values(image, image1)
        image, image1 = rescale(image, image1, max_ranges)

    g = np.uint8(difference_pixel(image, image1))
    name_to_archive = "Difference_of_"+name+"_"+str(variable1)+".png"
    ubication_final = MEDIA_ROOT + "/" + name_to_archive
    check_folder(ubication_final)
    cv.imwrite(ubication_final + "/" + name_to_archive, g)
    return True, name_to_archive, ubication_final


# multiplicación, division, blinding

def solve_dot(path, name, variable1):
    image = cv.imread(path + "/" + name)
    message_failed = "Didn't create file"
    image1 = variable1
    if type(variable1) != str:
        if image is None:
            return False, message_failed
    else:
        image1 = cv.imread(path + "/" + variable1)
        if image is None or image1 is None:
            return False, message_failed

        max_ranges = get_max_values(image, image1)
        image, image1 = rescale(image, image1, max_ranges)

    g = np.uint8(dot_images(image, image1))
    name_to_archive = "Dot_of_" + name + "_" + str(variable1) + ".png"
    ubication_final = MEDIA_ROOT + "/" + name_to_archive
    check_folder(ubication_final)
    cv.imwrite(ubication_final + "/" + name_to_archive, g)
    return True, name_to_archive, ubication_final


def solve_division(path, name, variable1):
    image = cv.imread(path + "/" + name)
    message_failed = "Didn't create file"
    image1 = variable1
    if type(variable1) != str:
        if image is None:
            return False, message_failed
    else:
        image1 = cv.imread(path + "/" + variable1)
        if image is None or image1 is None:
            return False, message_failed

        max_ranges = get_max_values(image, image1)
        image, image1 = rescale(image, image1, max_ranges)

    g = np.uint8(division_image(image, image1))
    name_to_archive = "Division_of_" + name + "_" + str(variable1) + ".png"
    ubication_final = MEDIA_ROOT + "/" + name_to_archive
    check_folder(ubication_final)
    cv.imwrite(ubication_final + "/" + name_to_archive, g)
    return True, name_to_archive, ubication_final


def solve_blinding(path, name, name1, variable1):
    image = cv.imread(path + "/" + name)
    message_failed = "Didn't create file"
    image1 = cv.imread(path + "/" + name1)
    if image is None or image1 is None:
        return False, message_failed

    max_ranges = get_max_values(image, image1)
    image, image1 = rescale(image, image1, max_ranges)

    g = np.uint8(blinding_image(image, image1, variable1))
    name_to_archive = "blinding_of_" + name + "_" + name1 + ".png"
    ubication_final = MEDIA_ROOT + "/" + name_to_archive
    check_folder(ubication_final)
    cv.imwrite(ubication_final + "/" + name_to_archive, g)
    return True, name_to_archive, ubication_final


def solve_and(path, name, name1):
    image = cv.imread(path + "/" + name)
    image1 = cv.imread(path + "/" + name1)
    message_failed = "Didn't create file"
    if image is None or image1 is None:
        return False, message_failed

    max_ranges = get_max_values(image, image1)
    img1, img2 = rescale(image, image1, max_ranges)
    g = np.uint8(and_operator(img1, img2))
    name_to_archive = "And_of_"+name+"_"+str(name1)+".png"
    ubication_final = MEDIA_ROOT + "/" + name_to_archive
    check_folder(ubication_final)
    cv.imwrite(ubication_final + "/" + name_to_archive, g)
    return True, name_to_archive, ubication_final


def solve_or(path, name, name1):
    image = cv.imread(path + "/" + name)
    image1 = cv.imread(path + "/" + name1)
    message_failed = "Didn't create file"
    if image is None or image1 is None:
        return False, message_failed

    max_ranges = get_max_values(image, image1)
    img1, img2 = rescale(image, image1, max_ranges)
    g = np.uint8(or_operator(img1, img2))
    name_to_archive = "OR_of_"+name+"_"+str(name1)+".png"
    ubication_final = MEDIA_ROOT + "/" + name_to_archive
    check_folder(ubication_final)
    cv.imwrite(ubication_final + "/" + name_to_archive, g)
    return True, name_to_archive, ubication_final


def solve_xor(path, name, name1):
    image = cv.imread(path + "/" + name)
    image1 = cv.imread(path + "/" + name1)
    message_failed = "Didn't create file"
    if image is None or image1 is None:
        return False, message_failed

    max_ranges = get_max_values(image, image1)
    img1, img2 = rescale(image, image1, max_ranges)
    g = np.uint8(xor_operator(img1, img2))
    name_to_archive = "XOR_of_"+name+"_"+str(name1)+".png"
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

    val = get_ranges_limits(get_histogram(image), constant, constant1)
    g = np.uint8(contrast_stretching_operator(image, val))
    name_to_archive = "Contrast_Streching_of_"+name+"_"+str(constant)+"_"+str(constant1)+".png"
    ubication_final = MEDIA_ROOT + "/" + name_to_archive
    check_folder(ubication_final)
    cv.imwrite(ubication_final + "/" + name_to_archive, g)
    return True, name_to_archive, ubication_final


def get_histogram_amount(img):
    f = plt.hist(img.ravel(), 256, [0, 256])
    return f[0]


def get_new_intensities(img, amount_bits=8):
    values = get_histogram_amount(img)
    cols, rows = img.shape
    length_pixels = 2**amount_bits
    probabilities = values / (cols*rows)
    intensities = np.array([0]*len(probabilities))
    accumulate = 0
    for i in range(len(probabilities)):
        accumulate += probabilities[i]
        intensities[i] = int((length_pixels - 1) * accumulate)
    return intensities


def solve_extra(img, sub_image=False, p_start1=None, p_end1=None):
    data = []
    if sub_image:
        blank_image = np.copy(img[p_start1[0]:p_end1[0] + 1, p_start1[1]:p_end1[1] + 1])
        data = get_new_intensities(blank_image)
    else:
        data = get_new_intensities(img)

    g = histogram_equalization(np.copy(img), data)
    return g


def solve_histogram_equalization(path, name, constant, constant1):
    image = cv.imread(path + "/" + name)
    message_failed = "Didn't create file"
    if image is None:
        return False, message_failed

    g = np.copy(image)
    indicator1 = True
    if constant == (0, 0) and constant1 == (0, 0):
        indicator1 = False

    for i in range(3):
        if indicator1:
            g[:, :, i] = solve_extra(g[:, :, i], indicator1, constant, constant1)
        else:
            g[:, :, i] = solve_extra(g[:, :, i])

    name_to_archive = "Histogram_equalization_of_"+name+"_"+str(constant)+"_"+str(constant1)+".png"
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


def up_image(my_file, name):
    ubication_image = MEDIA_ROOT + "/" + name
    fs = FileSystemStorage(location=ubication_image)
    filename = fs.save(my_file.name, my_file)
    return my_file.name
