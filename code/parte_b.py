import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from utilities import balanced_kernel, eliminate_non_maximum, convolution, find_by_percentage, binarize_img

# Parte b
kernel = balanced_kernel(8, -1, (3,3))
image = 'ChessBoard'
variation = 'peones'
image_path = f'imagenes/{image}.jpg'
fig_path = 'figures/'
threshold = 0.6
neighborhood = 20

# Se binariza la imagen y se grafica
binary_image = binarize_img(image_path)
fig, ax = plt.subplots()
ax.set(xlabel = 'x', ylabel = 'y', title = f'Resultado de la imagen binarizada')
ax.imshow(binary_image, cmap='binary')
plt.savefig(f'{fig_path}binaria_{image}.jpg')
plt.show()

# Se extrae un patron
fig, ax = plt.subplots()
ax.imshow(binary_image, cmap='binary')
points = []
plt.title('Selecciona 2 puntos, primero la esquina superior izquierda \ny luego la esquina inferior derecha')
points = np.asarray(plt.ginput(2, timeout=-1))
plt.title('Cierre este gráfico para continuar')
plt.draw()
plt.show()


pattern = binary_image[int(points[0,1]):int(points[1,1]), int(points[0,0]):int(points[1,0])]

# Se grafica el patron
fig, ax = plt.subplots()
ax.imshow(pattern, cmap='binary')
ax.set(xlabel = 'x', ylabel = 'y', title = f'Patron {variation} a detectar extraido de la imagen {image}')
plt.savefig(f'{fig_path}patron_{variation}_{image}.jpg')
plt.show()

# Se calcula el filtro detector para el patron
filter = convolution(pattern, kernel)
# Se realiza la prediccion
conv = convolution(binary_image, filter)


# Se muestran los resultados de la predicción
fig, ax = plt.subplots()
shw = ax.imshow(filter)
ax.set(xlabel = 'x', ylabel = 'y', title = f'Filtro calculado para el patron {variation} de la \nimagen {image}')
fig.colorbar(shw)
plt.savefig(f'{fig_path}filtro_{variation}_{image}.jpg')
plt.show()

fig, ax = plt.subplots()
shw = ax.imshow(conv)
ax.set(xlabel = 'x', ylabel = 'y', title = f'Convolución calculada con el filtro de {variation} y la \nimagen {image}')
fig.colorbar(shw)
plt.savefig(f'{fig_path}conv_{variation}_{image}.jpg')
plt.show()


# Se aplana la matriz de deteccion en un vector y se grafica
stimulus = conv.flatten()
fig, ax = plt.subplots()
ax.plot(stimulus)

ax.set(xlabel='Posición', ylabel='Estímulo',
       title=f'Detección al aplicar el filtro de {variation} sobre la imagen {image}')
ax.grid()
plt.savefig(f'{fig_path}estimulo_{variation}_{image}.jpg')
plt.show()

# Se filtran las posiciones de la imagen que tengan una prediccion mayor al threshold
indices =  find_by_percentage(conv, threshold)


w = pattern.shape[0]
h = pattern.shape[1]

x = indices[0]
y = indices[1]
fig, ax = plt.subplots()
ax.set(xlabel = 'x', ylabel = 'y', title=f'Detecciones de {variation} sin supresión de no máximos = {len(x)}')
plt.imshow(binary_image)

# Se grafican las predicciones sobre la imagen
for i in range(len(x)):
    ax.add_patch(patches.Rectangle((y[i],x[i]),h, w, fill=False, edgecolor='red', lw=1))

plt.savefig(f'{fig_path}deteccion_{variation}_sin_{image}.jpg')
plt.show()

indices = eliminate_non_maximum(indices, neighborhood)
x = indices[:, 0]
y = indices[:, 1]
fig, ax = plt.subplots()
ax.set(xlabel= 'x', ylabel ='y', title=f'Detecciones de {variation} con supresión de no máximos = {indices.shape[0]}')
plt.imshow(binary_image)
for i in range(len(x)):
    ax.add_patch(patches.Rectangle((y[i],x[i]),h, w, fill=False, edgecolor='red', lw=1))
plt.savefig(f'{fig_path}deteccion_{variation}_con_{image}.jpg')
plt.show()