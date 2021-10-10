from utilities import *
import numpy as np
import matplotlib.pyplot as plt

t = np.zeros((10,10))

t[3:7, 5] = 1
t[4,4:7] = 1
kernel = balanced_kernel(8, -1, (3,3))

kernel = np.fliplr(np.flipud(kernel))
filtro = convolution(t,kernel, False)
print(filtro)
#print(convolution(t,convolution(t, kernel, False),False))

fig, ax = plt.subplots()
plt.imshow(t)
plt.show()