import numpy as np

m = 8
n = 8
r = 3

V = np.random.rand(m,n) * 10
#print(V)
W =  np.random.rand(m,r)* 10
H =  np.random.rand(r,n)* 10

W_padded = np.hstack((W, np.zeros((m,4-r))))
H_padded = np.vstack((H, np.zeros((4-r, n))))

#print(np.transpose(W) @ W)
#print(np.transpose(W_padded) @ W_padded)

for i in range(2):
    H_num = np.transpose(W) @ V
    H_den = np.transpose(W) @ W @ H
    H_n1 = H*H_num/H_den
    W_num = V @ np.transpose(H_n1)
    W_den = W @ H_n1 @ np.transpose(H_n1)
    W_n1 = W*W_num/W_den
    print("Hn+1")
    print(H_n1)
    print("Wn+1")
    print(W_n1)

    H_num_padded = np.transpose(W_padded) @ V
    H_den_padded = np.transpose(W_padded) @ W_padded @ H_padded
    H_n1_padded = H_padded*H_num_padded/H_den_padded
    H_n1_padded = np.nan_to_num(H_n1_padded)
    
    W_num_padded = V @ np.transpose(H_n1_padded)
    W_den_padded = W_padded @ H_n1_padded @ np.transpose(H_n1_padded)
    W_n1_padded = W_padded*W_num_padded/W_den_padded
    W_n1_padded = np.nan_to_num(W_n1_padded)
    print("Hn+1 padded")
    print(H_n1_padded)
    print("Wn+1 padded")
    print(W_n1_padded)

    W_padded = W_n1_padded
    H_padded = H_n1_padded
    H = H_n1
    W = W_n1
    print("\n\n")
print(H_den_padded)

print(H_n1)
print()
print(H_n1_padded)




