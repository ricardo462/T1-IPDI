import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from utilities import eliminate_non_maximum
from utilities import convolution
from utilities import balanced_kernel
from utilities import find_by_percentage
from utilities import binarize_img



# Se crea una matriz de ceros de 10x10
l = np.zeros((10,10))
v = np.zeros((10,10))
# Se crea un un arreglo bidimensional que contiene una "L"
l[4:7, 4] = 1
l[7, 4:6] = 1

kernel = balanced_kernel(8, -1, (3,3))
filtro = convolution(l, kernel, False)
filtro = filtro[2:8, 2:6]
conv = convolution(l, filtro)


plt.subplot(211)
plt.imshow(filtro)
plt.subplot(212)
plt.imshow(conv)

plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes([0.85, 0.1, 0.075, 0.8])
plt.colorbar(cax=cax)
plt.show()

# Parte b
binary_image = binarize_img("imagenes/seis.jpg")


fig, ax = plt.subplots()
ax.imshow(binary_image, cmap='binary')


points = []
while len(points) < 2:
    points = np.asarray(plt.ginput(2, timeout=-1))
plt.show()

pattern = binary_image[int(points[0,1]):int(points[1,1]), int(points[0,0]):int(points[1,0])]

fig, ax = plt.subplots()
ax.imshow(pattern, cmap='binary')
plt.show()

diamond_filter = convolution(pattern, kernel, False)
conv = convolution(binary_image, diamond_filter, False)


plt.subplot(211)
plt.imshow(diamond_filter)
plt.subplot(212)
plt.imshow(conv)

plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes([0.85, 0.1, 0.075, 0.8])
plt.colorbar(cax=cax)
plt.show()

stimulus = conv.flatten()
fig, ax = plt.subplots()
ax.plot(stimulus)

ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax.grid()
plt.show()

indices =  find_by_percentage(conv,0.4)
x = indices[0]
y = indices[1]
w = pattern.shape[0]
h = pattern.shape[1]

fig, ax = plt.subplots()
plt.imshow(binary_image)
for i in range(len(x)):
    ax.add_patch(patches.Rectangle((y[i],x[i]),h, w, fill=False, edgecolor='red', lw=1))
plt.show()

print(indices[0].shape)

print(eliminate_non_maximum(indices).shape)

