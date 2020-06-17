import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys

def promedio(arr):
    filas, colum = arr.shape
    val = 0
    for i in range (filas):
        for k in range (colum):
            val = val + arr[i][k]
    prom = val / (filas*colum)

    return prom


def thresholding_adap(img1, c, ventana):
    filas, colum = img1.shape
    div = img1.copy()
    for i in range(filas):
        for k in range(colum):
            top = i - int((ventana - 1) / 2)
            down = i + int((ventana - 1) / 2)
            left = k - int((ventana - 1) / 2)
            right = k + int((ventana - 1) / 2)

            if top < 0:
                top = 0
            if down < filas:
                down = filas
            if left < 0:
                left = 0
            if right > colum:
                right = colum

            parte_img = img1[top:down, left:right]

            #umbral = promedio(parte_img) - c
            umbral = np.mean(parte_img) - c

            if img1[i, k] < umbral:
                div[i, k] = 0
            else:
                div[i, k] = 255

    #cv.imshow("prueba", div)
    return div

if __name__== "__main__":

    image1 = cv.imread("paper6.jpg", 0)

    if image1 is None:
        sys.exit("Can't read the image")

    g = []
    c = 2
    ventana = 11

    g = np.uint8(thresholding_adap(image1, c, ventana))

    cv.imshow("Main Image3", g)
    k = cv.waitKey(0)
    if k == ord("s"):
        cv.imwrite("Solution_img_1", g)

