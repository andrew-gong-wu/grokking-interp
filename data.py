import numpy as np

# for reproducibility
s = 15343489
np.random.seed(s)

# to store the data
N = 113
data = np.zeros((N**2, 4), dtype = int)

# generate 113^2 pairs 
for i in range(N):
    for j in range(N):
        # i + j = (i + j) mod N
        # we code = as N itself
        data[i*N + j] = [i,j,N, (i+j) % N]

# split into train and test
np.random.shuffle(data)
split = int(0.3 * N**2)
Xtr, Ytr = [_[0:3] for _ in data[:split]], [_[3] for _ in data[:split]]
Xte, Yte = [_[0:3] for _ in data[split:]], [_[3] for _ in data[split:]]

print(len(Xtr),len(Xte))
