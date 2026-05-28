import numpy as np
import torch

def generate_data(N=113, frac_train=0.3, s=15343489):
    # for reproducibility
    np.random.seed(s)

    # to store the data
    data = np.zeros((N**2, 4), dtype = int)

    # generate 113^2 pairs 
    for i in range(N):
        for j in range(N):
            # i + j = (i + j) mod N
            # we code = as N itself
            data[i*N + j] = [i,j,N, (i+j) % N]

    # split into train and test
    np.random.shuffle(data)
    split = int(frac_train * N**2)
    Xtr, Ytr = [_[0:3] for _ in data[:split]], [_[3] for _ in data[:split]]
    Xte, Yte = [_[0:3] for _ in data[split:]], [_[3] for _ in data[split:]]

    Xtr, Ytr, Xte, Yte = torch.tensor(Xtr), torch.tensor(Ytr), torch.tensor(Xte), torch.tensor(Yte)
    print(Xtr.shape)
    return Xtr, Ytr, Xte, Yte

generate_data()