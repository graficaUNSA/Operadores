import cv2 as cv
import numpy as np
import sys


def thresholding_operator(img, r1, r2):
    return np.where((r1 <= img) & (img <= r2), 0, 255)


def exponential_operator(constant, second_constant, data_pixel):
    return constant * (np.power(second_constant, data_pixel) - 1)


def contrast_stretching_operator(image_con, values, limit_inf=0, limit_sup=255):
    for i in range(3):
        diference = values[i][1]-values[i][0]
        image_con[:, :, i] = (image_con[:, :, i] - values[i][0]) * ((limit_sup-limit_inf)/diference) + limit_inf
    return image_con


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


def add_pixel(img1, img2):
    return img1 + img2


def difference_pixel(img1, img2):
    return np.abs(img1 - img2)


def rescale(img, img1, value):
    first_image = cv.resize(img, value)
    second_image = cv.resize(img1, value)
    return first_image, second_image


def get_max_values(img, img1):
    rows1 = max(img.shape[0], img1.shape[0])
    columns1 = max(img.shape[1], img1.shape[1])
    return rows1, columns1


image = cv.imread("sub_10.jpg")
image2 = cv.imread("sub_11.jpg")
state = False
if image is None or image2 is None:
    sys.exit("Can't read the image")

rows, columns = get_max_values(image, image2)
fst_image, scn_image = rescale(image, image2, (columns, rows))
g = []

if state:
    g = np.uint8(add_pixel(fst_image/2, scn_image/2))
else:
    constant = 0
    g = np.uint8(difference_pixel(fst_image, scn_image) + constant)
    cv.imshow("Main ImageStart", g)
    val = get_ranges_limits(get_histogram(g), 30, 95)
    g = np.uint8(contrast_stretching_operator(np.copy(g), val))




    # Estos valores son para las hojas
    # constant = 130
    # g = np.uint8(difference_pixel(fst_image, scn_image) + constant)
    # g = np.uint8(thresholding_operator(g, 0, 105))

cv.imshow("Main Image1", g)
k = cv.waitKey(0)
if k == ord("s"):
    cv.imwrite("Solution_img_1", g)
