import cv2
import numpy as np
from matplotlib import pyplot as plt

def son(L,n,Pon):
	L = L-1
	Pr = 0
	for i in range(0,n,1):
		Pr += Pon[i]
	return L*Pr

def Pon(L,size,pon):
    for x in L:
        pon.append(x/size)

imgc= cv2.imread('hist10_1.jpg')
img = cv2.cvtColor(imgc, cv2.COLOR_BGR2GRAY)

mini_img = img[200:200+110,172:172+88]

cv2.imshow('Mini imagen',mini_img)
cv2.waitKey(0)
plt.axis("on")

L = plt.hist(mini_img.ravel(),256,[0,256])[0]
print(L,"  ",len(L))
plt.show()

height, width = mini_img.shape
size = height * width

height, width = img.shape
print(size)

pon = []
Pon(L,size,pon)
new_sn = []

for i in range (1,len(L)+1):
	new_sn.append(int(son(len(L),i,pon)))

print("size: ",new_sn)
for y in range(0,width):
	for x in range(0,height):
		img[x,y] = new_sn[img[x,y]]
cv2.imshow('New Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.hist(img.ravel(),256,[0,256])[0]
plt.show()
