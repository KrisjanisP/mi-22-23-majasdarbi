import numpy as np


def dot(a, b):
    is_transposed = False

    a = np.atleast_2d(a)  # [1,2,3] -> [[1,2,3]]
    b = np.atleast_2d(b)  # [1,2,3] -> [[1,2,3]]

    if a.shape[1] != b.shape[0]:
        is_transposed = True
        b = np.transpose(b)

    a_rows = a.shape[0]
    b_columns = b.shape[1]

    product = np.zeros((a_rows, b_columns))

    for i in range(a_rows):
        for j in range(b_columns):
            product[i, j] += np.sum(a[i, :] * b[:, j])

    if is_transposed:
        product = np.transpose(product)

    if product.shape[0] == 1:
        product = product.flatten()

    return product


A = np.array([
    [1, 3, 6],
    [5, 2, 8]
])

B = np.array([
    [1, 3],
    [5, 2],
    [6, 9]
])
C = np.array([1, 2, 3])
D = np.array([1, 2])

print(dot(C, B), np.dot(C, B))
print(dot(B, D), np.dot(B, D))
