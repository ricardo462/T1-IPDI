import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from utilities import convolution
from utilities import balanced_kernel
from utilities import find_by_percentage
from PIL import Image


# Se crea una matriz de ceros de 10x10
l = np.zeros((10,10))
v = np.zeros((10,10))
# Se crea un un arreglo bidimensional que contiene una "L"
l[4:7, 4] = 1
l[7, 4:6] = 1

kernel = balanced_kernel(8, -1, (3,3))
filtro = convolution(l, kernel, False)
filtro = np.flipud(np.fliplr(filtro[2:8, 2:6]))
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

imagen = np.array(Image.open("imagenes/ChessBoard.jpg"))
dimensions = imagen.shape
binary_image = np.zeros((dimensions[0], dimensions[1]))
threshold = 255*3/2
for i in range(dimensions[0]):
    for j in range(dimensions[1]):
        binary_image[i, j] = 0 if np.sum(imagen[i, j, :]) > threshold else 1




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
diamond_filter = np.flipud(np.fliplr(diamond_filter))

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

fig.savefig("test.png")
plt.show()
"""
index = find(conv,0.7)
print(index)
x = index[0,:]
y = index[1,:]

x_distinct = []
for i in x:
    if i not in x_distinct:
        x_distinct.append(i)
y_distinct = []
for i in y:
    if i not in y_distinct:
        y_distinct.append(i)

print('[x]', x_distinct)
print('[y]', y_distinct)
"""

indices =  find_by_percentage(conv,0.5)
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

