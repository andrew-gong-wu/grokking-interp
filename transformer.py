import torch

# for reproducibility
g = torch.Generator().manual_seed(15343489)

# store relevant dimensions
d_embd = 128
d_attn_head = d_embd//4
n_hunit = 512
N = 113
num_chars = 114

# store embeddings
We = torch.randn((num_chars, d_embd), generator = g)/(d_embd**0.5)

# get positional embeddings
Wpos = torch.randn((3, d_embd), generator = g)/(d_embd**0.5)

# toy example
x = torch.tensor([[5, 94, 113],
                  [16,2,113],
                  [90,15,113],
                  [56,79,113]])
x_embd = We[x] + Wpos

print(x_embd.shape)

print(Wpos.shape)

# really, x will be 3800 x 3
# We[x] will be 3800 x 3 x 128
# in this toy example, x is 4x3 and We[x] is 4x3x128



# qkv projection
#Wq = torch.randn()

# our tokens are integers from 0 thru 112, and the equals sign (113)
# therefore there are 114 total token indices. 
