# -*- coding: utf-8 -*-
"""Histogram_equalization.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1w9G2vLoOqkGmisLHVz1EsDbPKo8d_PFC

Date: 03/01/2020
Author: Kritika Dhawale
"""

import pylab as plt
import matplotlib.image as mpimg
import numpy as np

import numpy as np 
#calculamos el histograma normalizado de una imagen
def imhist(im):
    m, n = im.shape
    h = [0.0] * 256
    for i in range(m):
        for j in range(n):
            h[im[i, j]]+=1
    return np.array(h)/(m*n)

#encontramos la suma acumulativa del array numpy
def cumsum(h):
    return [sum(h[:i+1]) for i in range(len(h))]

#calculamos el histograma ecualizado
def histeq(im):
    h = imhist(im)
    cdf = np.array(cumsum(h)) # funcion de distribucion acumulativa
    sk = np.uint8(255 * cdf)  # encotramos valores de la funcion de transferencia
    s1, s2 = im.shape
    Y = np.zeros_like(im)
    
    # aplicamos los valores transformados para cada pixel
    for i in range(0, s1):
        for j in range(0, s2):
            Y[i, j] = sk[im[i, j]]
    H = imhist(Y)
    
    #retornamos la iamgen transformada,  original y el nuevo histograma
    # y la funcion de transformacion
    return Y , h, H, sk

#  cargamos la imagen para numpy array
img1 = np.uint8(mpimg.imread('hist5.jpg')*255.0)
img2 = np.uint8(mpimg.imread('hist6.jpg')*255.0)

# convertimos a escala de grises
# hacemos  para canales individuales RGBA para imagenes sin escala de grises

img1 = np.uint8((0.2126* img1[:,:,0]) + np.uint8(0.7152 * img1[:,:,1]) + np.uint8(0.0722 * img1[:,:,2]))
nueva_img1, h, new_h, sk = histeq(img1)


img2 = np.uint8((0.2126* img2[:,:,0]) + np.uint8(0.7152 * img2[:,:,1]) + np.uint8(0.0722 * img2[:,:,2]))
nueva_img2, h, new_h, sk = histeq(img2)

#Ploteamos la imagen 1 
plt.subplot(121)
plt.imshow(img1)
plt.title('Imagen Original hist5')
plt.set_cmap('gray')

plt.subplot(122)
plt.imshow(nueva_img1)
plt.title('Imagen Ecualizada por el histograma')
plt.set_cmap('gray')
plt.show()

# histogramas
fig = plt.figure()
fig.add_subplot(221)
plt.plot(h)
plt.title('Histograma Orginal hist5') 

fig.add_subplot(222)
plt.plot(new_h)
plt.title('Histograma Ecualizado hist5')


#Ploateamos la imagen 2 
plt.subplot(121)
plt.imshow(img2)
plt.title('Imange Original hist6')
plt.set_cmap('gray')

plt.subplot(122)
plt.imshow(nueva_img2)
plt.title('Imagen Ecualizada por el Histograma ')
plt.set_cmap('gray')

fig = plt.figure()
fig.add_subplot(221)
plt.plot(h)
plt.title('Histograma Orginal hist6') 

fig.add_subplot(222)
plt.plot(new_h)
plt.title('Histograma Ecualizado hist6')


plt.show()

