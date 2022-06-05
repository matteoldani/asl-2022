from sklearn.decomposition import NMF
import time 
import numpy as np
input_sizes = range(208, 1808, 32)
print("m,seconds")

r = 16
for m in input_sizes:
    n = m
    X = np.random.random((m, m))
    model = NMF(n_components=16, init='random', random_state=0, solver = "mu", tol = 0, max_iter = 100, beta_loss='frobenius')
    start = time.perf_counter()
    W = model.fit_transform(X)
    finish = time.perf_counter()
    num_iterations = 100
    print(f"{m},{(finish - start)}")
