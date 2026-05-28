import torch
import torch.nn.functional as F
import torch.nn as nn

# for reproducibility
g = torch.Generator().manual_seed(15343489)


class Transformer(nn.Module):

    def __init__(self, d_embd=128, num_heads=4, d_head=32, n_hunit=512, num_chars=114):
        super().__init__()

        self.d_head = d_head
        self.num_heads = num_heads

        # embedding into 128-dimensional space - 114x128
        self.We = nn.Parameter(torch.randn((num_chars, d_embd), generator = g)/(d_embd**0.5))

        # positional embedding - 3x128
        self.Wpos = nn.Parameter(torch.randn((3, d_embd), generator = g)/(d_embd**0.5))

        # weight matrices for QKV - 4x32x128
        self.Wq = nn.Parameter(torch.randn((num_heads, d_head, d_embd), generator = g)/(d_embd**0.5))
        self.Wk = nn.Parameter(torch.randn((num_heads, d_head, d_embd), generator = g)/(d_embd**0.5))
        self.Wv = nn.Parameter(torch.randn((num_heads, d_head, d_embd), generator = g)/(d_embd**0.5))

        # weight matrix for output - 128x128
        self.Wo = nn.Parameter(torch.randn((num_heads * d_head, d_embd), generator = g)/(d_embd**0.5))

        # unembedding
        self.Wu = nn.Parameter(torch.randn((d_embd, num_chars), generator = g) / (num_chars**0.5))

        # to call MLP
        self.mlp = MLP(d_embd, n_hunit)

        # 3x3 LT mask
        # [0 -inf -inf]
        # [0    0 -inf]
        # [0    0    0]
        self.register_buffer('mask', torch.triu(torch.ones(3,3) * (-1e10), diagonal=1))

    def forward(self, X):

        # 3830x3x128
        X_embd = self.We[X] + self.Wpos

        # QKV projection - 3830x3x32x4
        Q = torch.einsum('abc, dec->abed', X_embd, self.Wq)
        K = torch.einsum('abc, dec->abed', X_embd, self.Wk)
        V = torch.einsum('abc, dec->abed', X_embd, self.Wv)

        # row softmax QK^T; apply mask - 3830x3x3x4
        attn = F.softmax(torch.einsum('abcd, aecd->abed', Q, K)/(self.d_head**0.5)
                 + self.mask[:,:,None], dim = 2)
        
        # multiply by V - 3830x3x32x4
        attn_head_out = torch.einsum('abcd, aced->abed', attn, V)

        # concatenate along the 32 dimension - 3830x3x128
        multi_head_out = torch.cat([attn_head_out[:,:,:,_] for _ in range(self.num_heads)], dim = 2)

        # transform to 3830x3x128
        attn_layer_out = torch.einsum('abc, cd->abd', multi_head_out, self.Wo)

        # residual connection - 3830x3x128
        X_embd = X_embd + attn_layer_out

        # mlp - 3830x3x128
        X_embd = X_embd + self.mlp(X_embd)

        # just get the logits for the third column, for the '=' character
        logits = torch.einsum('abc,cd->abd', X_embd, self.Wu)[:,2,:113] # the = sign is never going to be in Y

        return logits


class MLP(nn.Module):

    def __init__(self, d_embd=128, n_hunit=512):
        super().__init__()

        # weights and biases
        self.W1 = nn.Parameter(torch.randn((d_embd,n_hunit), generator = g) / (d_embd**0.5))
        self.b1 = nn.Parameter(torch.zeros(n_hunit))
        self.W2 = nn.Parameter(torch.randn((n_hunit,d_embd), generator = g) / (n_hunit**0.5))
        self.b2 = nn.Parameter(torch.zeros(d_embd))

        
    def forward(self, X_embd):
        
        # project into 512-dimensional space - 3830x3x512
        preact = torch.einsum('abc,cd->abd', X_embd, self.W1) + self.b1
        h = F.relu(preact)

        # project back into 128-dimensional space - 3830x3x128
        mlp_out = torch.einsum('abc,cd->abd', h, self.W2) + self.b2

        return mlp_out