import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys


def and_operator(img1,img2):
    out = np.bitwise_and(img1, img2)
    return out

def or_operator(img1,img2):
    out = np.bitwise_or(img1, img2)
    return out

def xor_operator(img1,img2):
    out = np.bitwise_xor(img1, img2)
    return out

def not_operator(img1):
    out = np.bitwise_not(img1)
    return out


def rescale(img, img1, value):
    first_image = cv.resize(img, value)
    second_image = cv.resize(img1, value)
    return first_image, second_image


def get_max_values(img, img1):
    rows1 = max(img.shape[0], img1.shape[0])
    columns1 = max(img.shape[1], img1.shape[1])
    return rows1, columns1

if __name__=='__main__':
    image1 = cv.imread('dog1.png')
    image2 = cv.imread('dog2.png')

    #image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    #image2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)


    if image1 is None or image2 is None:
            sys.exit("Can't read the image")

    rows, columns = get_max_values(image1, image2)
    fst_image, scn_image = rescale(image1, image2, (columns, rows))
    g = []
    #g = np.uint8(and_operator(fst_image,scn_image))
    #g = np.bitwise_or(image1, image2)
    g = np.uint8(or_operator(fst_image,scn_image))
    #g = np.uint8(xor_operator(fst_image,scn_image))
    #g = np.bitwise_xor(image1, image2)
    #g = np.bitwise_not(fst_image)
    #g = np.bitwise_not(scn_image)


    #dest_or = cv.bitwise_or(image2, image1, mask = None)

    cv.imshow("Main Image1", g)
    k = cv.waitKey(0)
    if k == ord("s"):
        cv.imwrite("Solution_img_1", g)
