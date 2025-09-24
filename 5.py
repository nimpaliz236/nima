import numpy as np

A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

C = np.array([[9, 10],
              [11, 12]])

Z = np.zeros_like(A)


T = np.block([
    [A, B, Z],
    [C, A, B],
    [Z, C, A]
])

print(T)


v = np.array([1, 2, 3, 4, 5, 6])


res_direct = T @ v
print(res_direct)


v1, v2, v3 = v[0:2], v[2:4], v[4:6]


res_block = np.concatenate([
    A @ v1 + B @ v2,
    C @ v1 + A @ v2 + B @ v3,
    C @ v2 + A @ v3
])

print(res_block)


print( np.allclose(res_direct, res_block))

