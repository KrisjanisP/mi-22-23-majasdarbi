import numpy as np

a = np.array([1,2,3,4,5])
a= np.expand_dims(a, axis=-1)
print(a**2)