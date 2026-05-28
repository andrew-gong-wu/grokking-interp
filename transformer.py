import torch
import torch.nn.functional as F
import torch.nn as nn

# for reproducibility
g = torch.Generator().manual_seed(15343489)


class Transformer(nn.Module):

    def __init__(self, d_embd=128, num_heads=4, d_head=32, n_hunit=512, num_chars=114):
        super().__init__()

        # embedding into 128-dimensional space - 114x128
        self.We = nn.parameter(torch.randn((num_chars, d_embd), generator = g)/(d_embd**0.5))

        # positional embedding - 3x128
        self.Wpos = nn.parameter(torch.randn((3, d_embd), generator = g)/(d_embd**0.5))

        # weight matrices for QKV - 4x32x128
        self.Wq = nn.parameter(torch.randn((num_heads, d_head, d_embd), generator = g)/(d_embd**0.5))
        self.Wk = nn.parameter(torch.randn((num_heads, d_head, d_embd), generator = g)/(d_embd**0.5))
        self.Wv = nn.parameter(torch.randn((num_heads, d_head, d_embd), generator = g)/(d_embd**0.5))

        # weight matrix for output - 128x128
        self.Wo = nn.parameter(torch.randn((num_heads * d_head, d_embd), generator = g)/(d_embd**0.5))


    def forward(self, X):

        # 3830x3x128
        X_embd = We[X] + Wpos

        # QKV projection - 3830x3x32x4
        Q = torch.einsum('abc, dec->abed', X_embd, Wq)
        K = torch.einsum('abc, dec->abed', X_embd, Wk)
        V = torch.einsum('abc, dec->abed', X_embd, Wv)

        # 3x3 LT mask
        # [0 -inf -inf]
        # [0    0 -inf]
        # [0    0    0]
        mask = torch.triu(torch.ones(3,3) * (-1e10), diagonal = 1)

        # row softmax QK^T; apply mask - 3830x3x3x4
        attn = F.softmax(torch.einsum('abcd, aecd->abed', Q, K)/(d_head**0.5)
                 + mask[:,:,None], dim = 2)
        
        # multiply by V - 3830x3x32x4
        attn_head_out = torch.einsum('abcd, aced->abed', attn, V)

        # concatenate along the 32 dimension - 3830x3x128
        multi_head_out = torch.cat([attn_head_out[:,:,:,_] for _ in range(num_heads)], dim = 2)

        # transform to 3830x3x128
        attn_layer_out = torch.einsum('abc, cd->abd', multi_head_out, Wo)

        # residual connection - 3830x3x128
        X_embd = X_embd + attn_layer_out


class MLP(nn.module):

    def __init__(self, d_embd=128, num_heads=4, d_head=32, n_hunit=512, num_chars=114):
        super().__init__()

        # weights and biases
        self.W1 = nn.parameter(torch.randn((d_embd,n_hunit), generator = g) / (d_embd**0.5))
        self.b1 = nn.parameter(torch.zeros(n_hunit))
        self.W2 = nn.parameter(torch.randn((n_hunit,d_embd), generator = g) / (n_hunit**0.5))
        self.b2 = nn.parameter(torch.zeros(d_embd))

        # unembedding
        self.Wu = nn.parameter(torch.randn((d_embd, num_chars), generator = g) / (num_chars**0.5))
    
    def forward(self, X_embd, Y):
        
        # project into 512-dimensional space - 3830x3x512
        preact = torch.einsum('abc,cd->abd', X_embd, W1) + b1
        h = F.relu(preact)

        # project back into 128-dimensional space - 3830x3x128
        mlp_out = torch.einsum('abc,cd->abd', h, W2) + b2
        X_embd = X_embd + mlp_out

        # unembed
        logits = torch.einsum('abc,cd->abd', X_embd, Wu)[:,2,:113] # the = sign is never going to be in Y
        nll_loss = F.cross_entropy(logits, Y)




# next is the MLP part
# currently at 5 x 3 x 128

# W1 = torch.randn((d_embd,n_hunit), generator = g) / (d_embd**0.5) # first weight
# b1 = torch.zeros(n_hunit) # first bias

# preact = torch.einsum('abc,cd->abd', X_embd, W1) + b1
# h = F.relu(preact)
# # currently at 5 x 3 x 512

# W2 = torch.randn((n_hunit,d_embd), generator = g) / (n_hunit**0.5) # second weight
# b2 = torch.zeros(d_embd) # second bias

# mlp_out = torch.einsum('abc,cd->abd', h, W2) + b2
# X_embd = X_embd + mlp_out
# # back at 5 x 3 x 128

# # unembed
# Wu = torch.randn((d_embd, num_chars), generator = g) / (num_chars**0.5) 
# logits = torch.einsum('abc,cd->abd', X_embd, Wu)[:,2,:113] # the = sign is never going to be in Y
# # we need the last element along the middle dimension for the 5x3x114

# nll_loss = F.cross_entropy(logits, Y)






# store relevant dimensions
d_embd = 128
num_heads = 4
d_head = d_embd//num_heads
n_hunit = 512
N = 113
num_chars = 114

# store embeddings
We = torch.randn((num_chars, d_embd), generator = g)/(d_embd**0.5)

# get positional embeddings
Wpos = torch.randn((3, d_embd), generator = g)/(d_embd**0.5)

# toy example
X = torch.tensor([[5, 94, 113],
                  [16,2,113],
                  [90,15,113],
                  [56,79,113],
                  [27,97,113]])
Y = torch.tensor([99, 18, 105, 22, 11])

X_embd = We[X] + Wpos

# X_embd is 5 x 3 x 128
# attention heads have dimension 32 and there are 4 attention heads total

# weight matrices
Wq = torch.randn((num_heads, d_head, d_embd), generator = g)/(d_embd**0.5)
Wk = torch.randn((num_heads, d_head, d_embd), generator = g)/(d_embd**0.5)
Wv = torch.randn((num_heads, d_head, d_embd), generator = g)/(d_embd**0.5)

# qkv projections
Q = torch.einsum('abc, dec->abed', X_embd, Wq)
K = torch.einsum('abc, dec->abed', X_embd, Wk)
V = torch.einsum('abc, dec->abed', X_embd, Wv)
# these tensors are 5 (# of train points) x 3 (# of chars per train point) x 32 (dimension of projection) x 4 (number of heads)

# computing attention and attention out
# four dimensions here:
# first - # of training points
# second - query position
# third - key position
# fourth - attention head
# mask looks like 
# [0 -inf -inf]
# [0    0 -inf]
# [0    0    0]
mask = torch.triu(torch.ones(3,3) * (-1e10), diagonal = 1)
attn = F.softmax(torch.einsum('abcd, aecd->abed', Q, K)/(d_head**0.5)
                 + mask[:,:,None], dim = 2)
# now 5 x 3 x 3 x 4. 
# V is 5 x 3 x 32 x 4
attn_head_out = torch.einsum('abcd, aced->abed', attn, V)
# get each 5 x 3 x 32 and line them up along the 32 dim
multi_head_out = torch.cat([attn_head_out[:,:,:,_] for _ in range(num_heads)], dim = 2)
# 5 x 3 x 128
# output projection 128 x 128
Wo = torch.randn((num_heads * d_head, d_embd), generator = g)/(d_embd**0.5)
attn_layer_out = torch.einsum('abc, cd->abd', multi_head_out, Wo)
# residual connection
X_embd = X_embd + attn_layer_out

# next is the MLP part
# currently at 5 x 3 x 128

W1 = torch.randn((d_embd,n_hunit), generator = g) / (d_embd**0.5) # first weight
b1 = torch.zeros(n_hunit) # first bias

preact = torch.einsum('abc,cd->abd', X_embd, W1) + b1
h = F.relu(preact)
# currently at 5 x 3 x 512

W2 = torch.randn((n_hunit,d_embd), generator = g) / (n_hunit**0.5) # second weight
b2 = torch.zeros(d_embd) # second bias

mlp_out = torch.einsum('abc,cd->abd', h, W2) + b2
X_embd = X_embd + mlp_out
# back at 5 x 3 x 128

# unembed
Wu = torch.randn((d_embd, num_chars), generator = g) / (num_chars**0.5) 
logits = torch.einsum('abc,cd->abd', X_embd, Wu)[:,2,:113] # the = sign is never going to be in Y
# we need the last element along the middle dimension for the 5x3x114

nll_loss = F.cross_entropy(logits, Y)







# we need to get back to 5 x 3 x 128

# really, x will be 3800 x 3
# We[x] will be 3800 x 3 x 128
# in this toy example, x is 4x3 and We[x] is 4x3x128



# qkv projection
#Wq = torch.randn()

# our tokens are integers from 0 thru 112, and the equals sign (113)
# therefore there are 114 total token indices. 
