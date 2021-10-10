import matplotlib.pyplot as plt
import numpy as np
from utilities import convolution
from utilities import balanced_kernel

# Se crea una matriz de ceros de 10x10
l = np.zeros((10,10))
v = np.zeros((10,10))
# Se crea un un arreglo bidimensional que contiene una "L"
l[4:7, 4] = 1
l[7, 4:6] = 1

kernel = balanced_kernel(-1, 8, (3,3))
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