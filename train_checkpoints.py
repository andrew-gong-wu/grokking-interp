# identical to train.py with the following caveats:
# 10000 epochs instead of 40000
# save loss for logits restricted to key frequencies and for logits restricted to non-key frequencies

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math

from data import generate_data
from transformer import Transformer 


Xtr, Ytr, Xte, Yte = generate_data()
model = Transformer()

# lr: 0.001
# weight decay=1.0: each parameter gets multiplied by (1 - lr*wd) = 0.999, which shrinks them
# betas: moving average of gradient/squared gradient
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))
# starts learning rate at 0 and ramps it up to 0.001 linearly
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step/10, 1))

# total number of epochs - reduce to 10000 because the model groks pretty early
M = 10000
N = 113


# we need to log the train and test losses
train_loss = []
test_loss = []

# log also the restricted / excluded lossees
restricted_loss = []
excluded_loss = []

# log the train and test accuracies
train_acc = []
test_acc = []

# create fourier basis, get all_data, key mask (code from analysis.ipynb)
# ------------------------------------------------------------
fourier_basis = []
for k in range(1,57):
    # each sin basis vector looks as follows: [sin((2kpi*0)/113), sin((2kpi*1)/113), ..., sin((2kpi*112)/113)]
    fourier_basis.append([math.sin(2 * k * math.pi * x / N) for x in range(N)])

for k in range(1,57):
    # each cos basis vector looks as follows: [cos((2kpi*0)/113), cos((2kpi*1)/113), ..., cos((2kpi*112)/113)]
    fourier_basis.append([math.cos(2 * k * math.pi * x / N) for x in range(N)])

# add the all-ones vector
fourier_basis.append([1] * N)
fourier_basis = torch.tensor(fourier_basis).float()

# normalize
fourier_basis = F.normalize(fourier_basis, dim = 1)

key_freqs = [13, 19, 35, 39, 49]
key_mask = torch.zeros(113)
for k in key_freqs:
    key_mask[k-1] = 1 # for sins
    key_mask[55+k] = 1 # for cosines   
key_mask[112] = 1 # for all-ones 

excluded_mask = 1 - key_mask
excluded_mask[112] = 1  # keep the constant for excluded too

# ------------------------------------------------------------

@torch.no_grad()
def split_loss_acc(split, model):

    # allows us to calculate loss for either train or test
    X,Y = {
        'train': (Xtr, Ytr),
        'test': (Xte, Yte)
    }[split]

    # calculates loss
    logits = model(X)
    loss = F.cross_entropy(logits, Y)

    # calculates accuracy
    # logits is Xlen x 113. we want the position of the maximum in the 113
    predictions = logits.argmax(dim = 1)
    accuracy = (predictions == Y).float().mean().item()

    return [loss.item(), accuracy]



for epoch in range(M):

    # forward 
    logits = model(Xtr)
    loss = F.cross_entropy(logits, Ytr)

    

    # backward
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    if epoch % 100 == 0:

        # log train and test losses
        train_loss_acc = split_loss_acc('train', model)
        test_loss_acc = split_loss_acc('test', model)
        train_loss.append(train_loss_acc[0])
        test_loss.append(test_loss_acc[0])
        train_acc.append(train_loss_acc[1])
        test_acc.append(test_loss_acc[1])

        # print losses for monitoring behavior
        print(f"Epoch {epoch}, train loss: {loss.item():.4f}")
        print(f"Epoch {epoch}, test loss: {test_loss_acc[0]:.4f}")

        with torch.no_grad():
            logits = model(Xte)
            coeffs = logits @ fourier_basis.T

            restricted = coeffs * key_mask
            excluded = coeffs * excluded_mask

            restricted_logits = restricted @ fourier_basis
            excluded_logits = excluded @ fourier_basis
        
            restricted_loss.append(F.cross_entropy(restricted_logits, Yte).item())
            excluded_loss.append(F.cross_entropy(excluded_logits, Yte).item())

    

# save loss and accuracy for plotting
# update to ensure it saves to relative filepath, not absolute one. had to dredge it up from somewhere
np.savez('logs_progress.npz', train_loss=train_loss, test_loss=test_loss,
            train_acc=train_acc, test_acc=test_acc,
            restricted_loss=restricted_loss, excluded_loss=excluded_loss)

# save model to save weights
torch.save(model.state_dict(), 'model.pth')
