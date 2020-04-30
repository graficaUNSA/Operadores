import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys


def convert2binary_image(img, r1, r2):
    cols, rows = img.shape
    for i in range(cols):
        for j in range(rows):
            if r1 <= img[i][j] <= r2:
                img[i][j] = 255
            else:
                img[i][j] = 0
    cv.imshow("Imagen transformada", img)


def solve(img):
    # value = img[246][18] #Punto usado para obtener el color de una celula viva en la imagen 1
    value = img[3][21] #Punto usado para obtener el color de una celula viva en la imagen 2
    # value = img[90][26] #Punto usado para obtener el color de una celula muerta(tono más claro) en la imagen 1
    # value = img[13][10] #Punto usado para obtener el color de una celula muerta(tono más claro) en la imagen 2
    #value = img[422][550] #punto usado para obtener el color del campo de trigo en la imagen 3
    print(value)
    """ 
    En el caso de la variable extra, su valor varia de acuerdo al pedido (cuando pide celulas vivas vale 2, en caso de celulas muertas vale 0 y
    en el caso de trigo tambien vale 0
    """
    extra = 2
    """
    En el caso del segundo parametro de la función convert2binary_image cuando se pide celulas muertas, el valor se
    coloca de acuerdo al histograma y la imagen (por ejemplo para la primera imagen vale 110 y para la segunda imagen
    vale 90)
    """
    convert2binary_image(img, value, value+extra)


def obtener_histograma(img):
    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()


def guardar_imagen(a, b=0):
    val = ""
    if a == 1:
        val += "Primera_imagen_"
    elif a == 2:
        val += "Segunda_imagen_"
    else:
        return "Tercera_imagen_campos_de_trigo"

    if b == 1:
        val += "Solo_vivas"
    else:
        val += "Solo_muertas"
    return val


original = cv.imread("thresh2.png")
img = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
if img is None:
    sys.exit("No se puede leer la imagen")


cv.imshow("Imagen original", img)
#obtener_histograma(img)
solve(img)
estado = 2
estado1 = 1
k = cv.waitKey(0)
if k == ord("s"):
    nombre = guardar_imagen(estado, estado1) + ".png"
    cv.imwrite(nombre, img)

