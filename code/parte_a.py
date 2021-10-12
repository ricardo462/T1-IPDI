import matplotlib.pyplot as plt
import numpy as np
from utilities import balanced_kernel, convolution

fig_path = 'figures/'
# Se crea una matriz de ceros de 10x10
l = np.zeros((10,10))
v = np.zeros((10,10))
# Se crea un un arreglo bidimensional que contiene una "L"
l[4:7, 4] = 1
l[7, 4:6] = 1

# Se crea un kernel balanceado de 3x3 con 8 al centro y -1 a los bordes
kernel = balanced_kernel(8, -1, (3,3))

# Se calculan el filtro y la convolucion con la imagen
filtro = convolution(l, kernel)
filtro = filtro[2:8, 2:6]
conv = convolution(l, filtro, True)


# Se grafican los resultados
fig, ax = plt.subplots()
ax.set(xlabel = 'x', ylabel = 'y', title = f'Imagen de L')
shw = ax.imshow(l)
plt.colorbar(shw)
plt.savefig(f'{fig_path}l.jpg')
plt.show()

fig, ax = plt.subplots()
ax.set(xlabel = 'x', ylabel = 'y', title = f'Filtro de L')
shw = ax.imshow(filtro)
plt.colorbar(shw)
plt.savefig(f'{fig_path}filtro_l.jpg')
plt.show()

fig, ax = plt.subplots()
ax.set(xlabel = 'x', ylabel = 'y', title = f'Convolución del filtro de L con la imagen')
shw = ax.imshow(conv)
plt.colorbar(shw)
plt.savefig(f'{fig_path}conv_l.jpg')
plt.show()