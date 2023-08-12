#!/usr/bin/env python
# coding: utf-8
# Bondoc Ion-Tudor, 333AA

# In[ ]:


import re
import cv2

#Obtin clasa si dimensiunile bounding box-ului din fisier pentru prima imagine
with open('bus1.txt', 'r') as file:
    content = file.read()
    info = re.findall(r"[-+]?(?:\d*\.*\d+)", content)
    classimg = info[0]
    xmij = info[1]
    ymij = info[2]
    width = info[3]
    height = info[4]
file.close()

#Obtin dimensiunile pentru crop
dim = 512 #dimensiunea unei imagini
xmij = float(xmij)
ymij = float(ymij)
width = float(width)
height = float(height)
x1 = (xmij - (width/2))*dim
x2 = (xmij + (width/2))*dim
y1 = (ymij - (height/2))*dim
y2 = (ymij + (height/2))*dim
print(x1, y1, x2, y2)
x1 = int(x1)
x2 = int(x2)
y1 = int(y1)
y2 = int(y2)
print(x1, y1, x2, y2)

#Incep lucrul cu imagini
img = cv2.imread('bus1.bmp')
cropped = img[y1:y2, x1:x2]
cv2.imwrite('bus1cropped.bmp', cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


#Obtin clasa si dimensiunile bounding box-ului din fisier pentru a doua imagine
with open('bus2.txt', 'r') as file:
    content = file.read()
    info = re.findall(r"[-+]?(?:\d*\.*\d+)", content)
    classimg = info[0]
    xmij = info[1]
    ymij = info[2]
    width = info[3]
    height = info[4]
file.close()

#Obtin dimensiunile pentru crop
dim = 512 #dimensiunea unei imagini
xmij = float(xmij)
ymij = float(ymij)
width = float(width)
height = float(height)
x1 = (xmij - (width/2))*dim
x2 = (xmij + (width/2))*dim
y1 = (ymij - (height/2))*dim
y2 = (ymij + (height/2))*dim
print(x1, y1, x2, y2)
x1 = int(x1)
x2 = int(x2)
y1 = int(y1)
y2 = int(y2)
print(x1, y1, x2, y2)

#Incep lucrul cu imagini
img = cv2.imread('bus2.bmp')
cropped = img[y1:y2, x1:x2]
cv2.imwrite('bus2cropped.bmp', cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


#Obtin clasa si dimensiunile bounding box-ului din fisier pentru a treia imagine
with open('bus3.txt', 'r') as file:
    content = file.read()
    info = re.findall(r"[-+]?(?:\d*\.*\d+)", content)
    classimg = info[0]
    xmij = info[1]
    ymij = info[2]
    width = info[3]
    height = info[4]
file.close()

#Obtin dimensiunile pentru crop
dim = 512 #dimensiunea unei imagini
xmij = float(xmij)
ymij = float(ymij)
width = float(width)
height = float(height)
x1 = (xmij - (width/2))*dim
x2 = (xmij + (width/2))*dim
y1 = (ymij - (height/2))*dim
y2 = (ymij + (height/2))*dim
print(x1, y1, x2, y2)
x1 = int(x1)
x2 = int(x2)
y1 = int(y1)
y2 = int(y2)
print(x1, y1, x2, y2)

#Incep lucrul cu imagini
img = cv2.imread('bus3.bmp')
cropped = img[y1:y2, x1:x2]
cv2.imwrite('bus3cropped.bmp', cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


import numpy as np

#poza 1

#Aplic masca folosind HSV
img = cv2.imread('bus1cropped.bmp')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#am gasit valorile pentru lower purple si higher purple cu ajutorul unei mape de culori HSV
lower_purple = np.array([130, 50, 50])
upper_purple = np.array([170, 255, 255])
mask = cv2.inRange(hsv, lower_purple, upper_purple)

#obtin si salvez imaginea cu masca aplicata
result = cv2.bitwise_and(img, img, mask=mask)
cv2.imwrite('bus1masked.bmp', result)

#convertesc rezultatul din BGR in formatul CIE Luv
result = cv2.cvtColor(result, cv2.COLOR_BGR2Luv)
cv2.imwrite('bus1maskedLUV.bmp', result)


# In[ ]:


#poza 2

#Aplic masca folosind HSV
img = cv2.imread('bus2cropped.bmp')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_purple = np.array([130, 50, 50])
upper_purple = np.array([170, 255, 255])
mask = cv2.inRange(hsv, lower_purple, upper_purple)

#obtin si salvez imaginea cu masca aplicata
result = cv2.bitwise_and(img, img, mask=mask)
cv2.imwrite('bus2masked.bmp', result)

#convertesc rezultatul din BGR in formatul CIE Luv
result = cv2.cvtColor(result, cv2.COLOR_BGR2Luv)
cv2.imwrite('bus2maskedLUV.bmp', result)


# In[ ]:


#poza 3

#Aplic masca folosind HSV
img = cv2.imread('bus3cropped.bmp')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_purple = np.array([130, 50, 50])
upper_purple = np.array([170, 255, 255])
mask = cv2.inRange(hsv, lower_purple, upper_purple)

#obtin si salvez imaginea cu masca aplicata
result = cv2.bitwise_and(img, img, mask=mask)
cv2.imwrite('bus3masked.bmp', result)

#convertesc rezultatul din BGR in formatul CIE Luv
result = cv2.cvtColor(result, cv2.COLOR_BGR2Luv)
cv2.imwrite('bus3maskedLUV.bmp', result)


# In[ ]:


#Transformare in masca cu transparenta
def bmp_to_png_with_transparency(image, alpha_channel):
    #creez o imagine goala neagra de dimensiunea imaginii, dar cu 4 canale
    output = np.zeros((image.shape[0], image.shape[1], 4))
    #atribui pixelii precum si cele 3 canale initiale
    output[:,:,0:3] = image.copy()
    #atribui al 4-lea canal
    output[:,:,3] = alpha_channel
    output = output.astype(np.uint8)
    #gasesc pixelii negri (B=0) si le aplic transparenta maxima
    output[output[:, :, 0] == 0, 3] = 0
    return output


# In[ ]:


#Aplicare transparenta pentru imaginea 1
bmp_image = cv2.imread('bus1masked.bmp')
cv2.imwrite('bus1maskedpng.png', bmp_image)
png_image = cv2.imread('bus1maskedpng.png', cv2.IMREAD_UNCHANGED)
png_image = cv2.cvtColor(png_image, cv2.COLOR_BGR2BGRA)
output = bmp_to_png_with_transparency(bmp_image, png_image[:,:,3])
cv2.imwrite('bus1transparent.png', output)


# In[ ]:


#Aplicare transparenta pentru imaginea 2
bmp_image = cv2.imread('bus2masked.bmp')
cv2.imwrite('bus2maskedpng.png', bmp_image)
png_image = cv2.imread('bus2maskedpng.png', cv2.IMREAD_UNCHANGED)
png_image = cv2.cvtColor(png_image, cv2.COLOR_BGR2BGRA)
output = bmp_to_png_with_transparency(bmp_image, png_image[:,:,3])
cv2.imwrite('bus2transparent.png', output)


# In[ ]:


#Aplicare transparenta pentru imaginea 3
bmp_image = cv2.imread('bus3masked.bmp')
cv2.imwrite('bus3maskedpng.png', bmp_image)
png_image = cv2.imread('bus3maskedpng.png', cv2.IMREAD_UNCHANGED)
png_image = cv2.cvtColor(png_image, cv2.COLOR_BGR2BGRA)
output = bmp_to_png_with_transparency(bmp_image, png_image[:,:,3])
cv2.imwrite('bus3transparent.png', output)


# In[ ]:




