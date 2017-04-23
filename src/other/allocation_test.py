import numpy as np

b, n = 2, 4
u = np.random.rand(b, n)
sorted_usage = np.sort(u, axis=1)
s = np.argsort(u, axis=1)

pw = np.zeros((b, n)).astype(np.float32)
for i in range(b):
    for j in range(n):
        product_list = [u[i, s[i, k]] for k in range(j)]
        pw[i, s[i, j]] = (1 - u[i, s[i, j]]) * np.product(product_list)

np.set_printoptions(precision=3, suppress=True)
mw = np.zeros((b, n)).astype(np.float32)
for i in range(b):
    cp = np.concatenate([[1], np.cumprod(u[i][s[i]])[:-1]])
    mw[i][s[i]] = (1 - u[i][s[i]]) * cp
print(u, end="\n---\n")
print(pw, end="\n---\n")
print(mw)
