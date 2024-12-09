from DL_functions import *
import torch
import torch.optim as optim
import numpy as np


# load data
Z = np.loadtxt("realZ_sample.txt")
R1 = np.loadtxt("realstock_return.txt")
R2 = np.loadtxt("realportfolio_return.txt")
M = np.loadtxt("realMKT.txt")
T = M.shape[0] # number of periods

data_input = dict(characteristics=torch.tensor(Z, dtype=torch.float32),
                  stock_return=torch.tensor(R1, dtype=torch.float32),
                  target_return=torch.tensor(R2, dtype=torch.float32),
                  factor=torch.tensor(M[:, 0:3], dtype=torch.float32))

# set parameters
training_para = dict(epoch=50, train_ratio=0.8, train_algo=optim.Adam,  
                     split="future", activation=torch.tanh, start=1, batch_size=120, learning_rate=0.05,
                     Lambda=1, Lambda2=1)

# design network layers
layer_size = [64, 32, 16, 8, 1]

# construct deep factors
f, char = dl_alpha(data_input, layer_size, training_para)
print(f, char)
