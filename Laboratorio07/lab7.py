import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys


def info():
    print("Select the operator: ")
    print("1. Dot two images")
    print("2. Dot of image per a constant")
    print("3. Division of two images")
    print()


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


def image_dot_const(img1,const):
    #const = float(input("Enter constant: "))
    return img1 * const


def  dot_images(img1,img2):
    return img1 * imag2


def image_div_const(img1):
    const = float(input("Eneter constant: "))
    return img1/const

def division_image(img1,img2):
    return img1/img2

def blending(img1,img2):
    const = float(input("Enter constant [0 - 1]: "))
    if const>=0 and const <=1:
        blen = const*img1 + (1-const)*img2
        return blen

def get_histogram(img):
    data = []
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv.calcHist([img], [i], None, [256], [0, 256])
        data += [get_ranges(histr)]
    plt.plot(histr, color='gray' )

    plt.xlabel('intensidad de iluminacion')
    plt.ylabel('cantidad de pixeles')
    plt.show()
    print(data)
    return data

def get_ranges_limits(values, l1, l2):
    nvalues = []
    for i in range(len(values)):
        amount = values[i][1] - values[i][0]
        individual_value = amount / 100
        nvalues += [(int(values[i][0] + l1*individual_value), int(values[i][1] - (100-l2) * individual_value))]
    return nvalues

def rescale(img, img1, value):
    first_image = cv.resize(img, value)
    second_image = cv.resize(img1, value)
    return first_image, second_image


def get_max_values(img, img1):
    rows1 = max(img.shape[0], img1.shape[0])
    columns1 = max(img.shape[1], img1.shape[1])
    return rows1, columns1


if __name__=="__main__":

    image1 = cv.imread("add_10.jpg") #catedral
    image2 = cv.imread("tiger.png") #leon

    #image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    #image2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

    if image1 is None or image2 is None:
        sys.exit("Can't read the image")

    rows, columns = get_max_values(image1, image2)
    fst_image, scn_image = rescale(image1, image2, (columns, rows))
    g = []

    const = 30
    #dot image per a constant - ejercicio 1
    #g = np.uint8(image_dot_const(fst_image))

    '''
    #ejercicio 3
    g = np.uint8(division_image(fst_image,scn_image))
    get_histogram(g)
    g = np.uint8(image_dot_const(np.copy(g),const))

    get_histogram(g)

    val = get_ranges_limits(get_histogram(g),0, 30)
    g = np.uint8(contrast_stretching_operator(np.copy(g), val))
    '''

    #ejercicio 4
    g = np.uint8(blending(fst_image,scn_image))


    cv.imshow("Main Image1", g)
    k = cv.waitKey(0)
    if k == ord("s"):
        cv.imwrite("Solution_img_1", g)
