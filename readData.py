""" read the data in csv format and display as an image """
import numpy as np
from matplotlib import pyplot as plt
data = np.genfromtxt('train.csv',delimiter=',',skip_header=1, max_rows=2)
img = np.reshape(data[0][1:], newshape=(28,28))
plt.imshow(img, interpolation='nearest', cmap='gray')
plt.show()
