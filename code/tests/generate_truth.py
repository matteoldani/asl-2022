from sklearn.decomposition import non_negative_factorization
import numpy as np

input_files = "./inputs/"
output_files= "./output_files/"
ground_truths= "./ground_truths/"

np.random.seed(42)

def generate_factorization(m, n, r,max_val = 1000):
    V = generate_matrix(m, n, max_val=max_val)
    W = generate_matrix(m,r, max_val = max_val)
    H = generate_matrix(r,n, max_val = max_val)
    np.savetxt(input_files + "V_nnm.matrix", V, fmt="%.12f", header = f"{V.shape[0]} {V.shape[1]}",comments = "")
    np.savetxt(input_files + "W_nnm_init.matrix", W, fmt="%.12f", header = f"{W.shape[0]} {W.shape[1]}",comments = "")
    np.savetxt(input_files + "H_nnm_init.matrix", H, fmt="%.12f", header = f"{H.shape[0]} {H.shape[1]}",comments = "")
    W, H, n_iter = non_negative_factorization(V, W, H, init = "custom",n_components=r, update_H=True, solver='mu',
                               beta_loss='frobenius', tol=1e-13,
                               max_iter=1000)
    np.savetxt(input_files + "W_nnm.matrix", W, fmt="%.12f", header = f"{W.shape[0]} {W.shape[1]}",comments = "")
    np.savetxt(input_files + "H_nnm.matrix", H, fmt="%.12f", header = f"{H.shape[0]} {H.shape[1]}",comments = "")


def generate_mul(m, n):
    A = generate_matrix(m,n)
    B = generate_matrix(n, m)
    R = np.matmul(A, B)
    np.savetxt(input_files + "/A_mul.matrix", A, fmt="%.12f", header = f"{A.shape[0]} {A.shape[1]}",comments = "")
    np.savetxt(input_files + "/B_mul.matrix", B, fmt="%.12f", header = f"{B.shape[0]} {B.shape[1]}",comments = "")
    np.savetxt(input_files + "/R_mul.matrix", R, fmt="%.12f", header = f"{R.shape[0]} {R.shape[1]}",comments = "")


def generate_ltrans_mul(m, n):
    A = generate_matrix(n,m)
    B = generate_matrix(n, m)
    R = np.matmul(np.transpose(A), B)
    np.savetxt(input_files + "/A_ltrans_mul.matrix", A, fmt="%.12f", header = f"{A.shape[0]} {A.shape[1]}",comments = "")
    np.savetxt(input_files + "/B_ltrans_mul.matrix", B, fmt="%.12f", header = f"{B.shape[0]} {B.shape[1]}",comments = "")
    np.savetxt(input_files + "/R_ltrans_mul.matrix", R, fmt="%.12f", header = f"{R.shape[0]} {R.shape[1]}",comments = "")


def generate_rtrans_mul(m, n):
    A = generate_matrix(m,n)
    B = generate_matrix(m, n)
    R = np.matmul(A, np.transpose(B))
    np.savetxt(input_files + "/A_rtrans_mul.matrix", A, fmt="%.12f", header = f"{A.shape[0]} {A.shape[1]}",comments = "")
    np.savetxt(input_files + "/B_rtrans_mul.matrix", B, fmt="%.12f", header = f"{B.shape[0]} {B.shape[1]}",comments = "")
    np.savetxt(input_files + "/R_rtrans_mul.matrix", R, fmt="%.12f", header = f"{R.shape[0]} {R.shape[1]}",comments = "")





def generate_matrix(r,c, max_val = 1000):
    return np.random.rand(r,c) * max_val

r = 400
c = 200

generate_mul(r, c)
generate_ltrans_mul(r, c)
generate_rtrans_mul(r, c)
generate_factorization(400, 400, 10)