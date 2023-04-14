# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 17:55:09 2021

@author: James
"""

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
from Network import *
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from datetime import datetime


use_cuda=True

if use_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
#%%hyperparams


d_model=20
d_ff=80
heads = 4
input_size=1
output_size=1
enc_seq_len = 7   # input_sequence_length
dec_seq_len = 5    # output_sequence_length
max_len=100

lr = 0.002
epochs = 20
n_layers=3
batch_size = 2

#init network and optimizer
transformer = Transformer(d_model,d_ff,heads,input_size,output_size,enc_seq_len,dec_seq_len,
                max_len,n_layers=n_layers).to(device=device)


optimizer = torch.optim.Adam(transformer.parameters(), lr=lr)

#keep track of loss for graph
losses = []

path='./log/' + datetime.now().strftime('%Y%m%d-%H%M%S')
writer = SummaryWriter(path)
X, Y = get_training_data(batch_size, enc_seq_len, dec_seq_len)
X=X.to(device=device)   # torch.Size([100, 10, 1])
Y=Y.to(device=device)   # torch.Size([20, 10])
mask_out=subsequent_mask(Y.shape[0],Y.shape[0]).to(device=device)       # torch.Size([20, 20, 1])
mask_enc=subsequent_mask(Y.shape[0],X.shape[0]).to(device=device)       # torch.Size([20, 100, 1])
Y1 = Y.unsqueeze(-1)        # torch.Size([20, 10, 1])
writer.add_graph(transformer, (X,Y.unsqueeze(-1),mask_out,mask_enc))
#%%training
    
for e in range(epochs):
    out = []
    
    for b in range(20):
        optimizer.zero_grad()
        #[seq_len,batch,feature_size]
        X, Y = get_training_data(batch_size, enc_seq_len, dec_seq_len)
        X=X.to(device=device)   # torch.Size([100, 10, 1])
        Y=Y.to(device=device)   # torch.Size([20, 10])
        mask_out=subsequent_mask(Y.shape[0],Y.shape[0]).to(device=device)   # torch.Size([20, 20, 1])
        mask_enc=subsequent_mask(Y.shape[0],X.shape[0]).to(device=device)   # torch.Size([20, 100, 1])
        
        #Forward pass and calculate loss
        net_out = transformer(X,Y.unsqueeze(-1),mask_out,mask_enc)
        #print(net_out.shape,Y.shape)
        loss = torch.mean((net_out - Y) ** 2)

        #backwards pass
        loss.backward()
        optimizer.step()

        #Track losses and draw rgaph
        out.append([net_out.cpu().detach().numpy(), Y])
        losses.append(loss.item())
        writer.add_scalar("training loss",losses[-1],e*20+b)

        print(f'epoch:{e+1}|step:{b+1}|loss:{losses[-1]}')


#%%testing

fig = plt.figure()
ax = fig.add_subplot(111)
plt.ion()

fig.show()
fig.canvas.draw()


X, Y = get_testing_data(enc_seq_len, dec_seq_len)

#Draw graph comparing to sigmoid
for i in range(1):
    
    X=X.to(device=device)
    Y=Y.to(device=device)
    mask_out=subsequent_mask(Y.shape[0],Y.shape[0]).to(device=device)
    mask_enc=subsequent_mask(Y.shape[0],X.shape[0]).to(device=device)

    outputs=transformer(X,Y.unsqueeze(-1),mask_out,mask_enc).cpu().detach().squeeze().numpy()


Y=Y.cpu().detach().squeeze().numpy()
ax.clear()
ax.plot(outputs, label='Network output')
ax.plot(Y, label='Sigmoid function')
ax.set_title("")
ax.legend(loc='upper left', frameon=False)



"""

fig = plt.figure()
ax = fig.add_subplot(111)
plt.ion()

fig.show()
fig.canvas.draw()

o = []
x = torch.sigmoid(torch.linspace(-10,0,enc_seq_len+1).float()).unsqueeze(-1).numpy().tolist()
Y = torch.ones((dec_seq_len, 1, output_size))*1e-15
Y = Y.to(device=device)


#Draw graph comparing to sigmoid
for i in range(dec_seq_len-1):
    o.append([torch.sigmoid(torch.tensor(i).float())])
    q = torch.tensor(x).float().unsqueeze(-1)
    
    X=q[-enc_seq_len-1:-1,:,:]
    X=X.to(device=device)

    Y[i, :, :] = q[-1,:,:]
    mask_out=subsequent_mask(Y.shape[0],Y.shape[0]).to(device=device)
    mask_enc=subsequent_mask(Y.shape[0],X.shape[0]).to(device=device)
    
    if(Y.shape[0] == 1):
        x.append([float(transformer(X,Y,mask_out,mask_enc).cpu().detach().squeeze().numpy())])
    else:
        x.append([float(transformer(X,Y,mask_out,mask_enc).cpu().detach().squeeze().numpy()[i+1])])


#还剩下一个没有写到Y里面
Y[i+1, :, :] = q[-1,:,:]
Y=Y.cpu().detach().squeeze().numpy()
ax.clear()
ax.plot(Y, label='Network output')
ax.plot(o, label='Sigmoid function')
ax.set_title("")
ax.legend(loc='upper left', frameon=False)

"""