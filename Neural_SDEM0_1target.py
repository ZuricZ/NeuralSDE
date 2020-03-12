import torch
import torch.nn as nn
import numpy as np
import math
import os
import time
from random import randrange

if torch.cuda.is_available():
    device = 'cuda'
    # Uncomment below to pick particular device if running on a cluster:
    torch.cuda.set_device(7)
    device = 'cuda:7'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.cuda.current_device()

else:
    device = 'cpu'
    torch.set_default_tensor_type('torch.FloatTensor')


class Net_timestep(nn.Module):
    def __init__(self, dim, nOut, n_layers, vNetWidth, activation="relu"):
        super(Net_timestep, self).__init__()
        self.dim = dim
        self.nOut = nOut

        if activation != "relu" and activation != "tanh":
            raise ValueError("unknown activation function {}".format(activation))
        if activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()

        self.i_h = self.hiddenLayerT1(dim, vNetWidth)
        self.h_h = nn.ModuleList([self.hiddenLayerT1(vNetWidth, vNetWidth) for l in range(n_layers - 1)])
        self.h_o = self.outputLayer(vNetWidth, nOut)

    def hiddenLayerT0(self, nIn, nOut):
        layer = nn.Sequential(  # nn.BatchNorm1d(nIn, momentum=0.1),
            nn.Linear(nIn, nOut, bias=True),
            # nn.BatchNorm1d(nOut, momentum=0.1),
            self.activation)
        return layer

    def hiddenLayerT1(self, nIn, nOut):
        layer = nn.Sequential(nn.Linear(nIn, nOut, bias=True),
                              # nn.BatchNorm1d(nOut, momentum=0.1),
                              self.activation)
        return layer

    def outputLayer(self, nIn, nOut):
        layer = nn.Sequential(nn.Linear(nIn, nOut, bias=True))
        return layer

    def forward(self, S):
        h = self.i_h(S)
        for l in range(len(self.h_h)):
            h = self.h_h[l](h)
        output = self.h_o(h)
        return output


# Set up neural SDE class

class Net_SDE(nn.Module):

    def __init__(self, dim, timegrid, strikes_call, n_layers, vNetWidth, device):

        super(Net_SDE, self).__init__()
        self.dim = dim
        self.timegrid = timegrid
        self.device = device
        self.strikes_call = strikes_call
        #     self.strikes_put = strikes_put

        # Input to each coefficient (NN) will be (t,S_t,V_t)

        self.diffusion = Net_timestep(dim=dim + 2, nOut=1, n_layers=n_layers, vNetWidth=vNetWidth)
        self.driftV = Net_timestep(dim=dim + 1, nOut=1, n_layers=n_layers, vNetWidth=vNetWidth)
        self.diffusionV = Net_timestep(dim=dim + 1, nOut=1, n_layers=n_layers, vNetWidth=vNetWidth)
        self.diffusionV1 = Net_timestep(dim=dim + 1, nOut=1, n_layers=n_layers, vNetWidth=vNetWidth)

        self.control_variate = Net_timestep(dim=dim, nOut=1, n_layers=4, vNetWidth=100)

    def forward(self, S0, V0, rate, BS_vol, indices, z, z1, MC_samples):
        S_old = torch.repeat_interleave(S0, MC_samples, dim=0)
        # Uncomment when using BS Control Variate:  
        # BS_old = torch.repeat_interleave(S0, MC_samples, dim=0)
        V_old = torch.repeat_interleave(V0, MC_samples, dim=0)
        K_call = self.strikes_call
        # K_put = self.strikes_put
        zeros = torch.repeat_interleave(torch.zeros(1, 1), MC_samples, dim=0)
        average_SS = torch.Tensor()
        average_SS1 = torch.Tensor()
        average_SS_OTM = torch.Tensor()
        average_SS1_ITM = torch.Tensor()
        # use fixed step size
        h = self.timegrid[1] - self.timegrid[0]
        n_steps = len(self.timegrid) - 1
        # set maturity counter
        countmat = -1

        # Control Variate
        cv = 0

        # Solve for S_t, V_t (Euler)   
        irand = [randrange(0, n_steps + 1, 1) for k in range(300)]
        for i in range(1, len(self.timegrid)):

            dW = (torch.sqrt(h) * z[:, i - 1]).reshape(MC_samples, 1)
            dW1 = (torch.sqrt(h) * z1[:, i - 1]).reshape(MC_samples, 1)
            current_time = torch.ones(1, 1) * self.timegrid[i - 1]
            input_time = torch.repeat_interleave(current_time, MC_samples, dim=0)
            inputNN = torch.cat([input_time.reshape(MC_samples, 1), S_old, V_old], 1)
            inputNNvol = torch.cat([input_time.reshape(MC_samples, 1), V_old], 1)

            input_CV = torch.cat([input_time.reshape(MC_samples, 1), S_old], 1)
            input_CV = S_old
            cv += self.control_variate(input_CV.detach()) * dW

            if int(i) in irand:
                S_new = S_old + S_old * rate * h + self.diffusion(inputNN) * dW
                V_new = V_old + self.driftV(inputNNvol) * h + self.diffusionV(inputNNvol) * dW + self.diffusionV1(
                    inputNNvol) * dW1
            else:
                S_new = S_old + S_old * rate * h + self.diffusion(inputNN).detach() * dW
                V_new = V_old + self.driftV(inputNNvol).detach() * h + self.diffusionV(
                    inputNNvol).detach() * dW + self.diffusionV1(inputNNvol).detach() * dW1
            S_new = torch.cat([S_new, zeros], 1)
            S_new = torch.max(S_new, 1, keepdim=True)[0]
            S_old = S_new
            V_old = V_new

            # If particular timestep is a maturity for Vanilla option

            if int(i) in indices:
                countmat += 1
                Z_new = torch.Tensor()
                Z_newP_ITM = torch.Tensor()
                Z_newP_OTM = torch.Tensor()
                Z_new2 = torch.Tensor()
                # countstrikecall=-1

                # Evaluate put (OTM) and call (OTM) option prices 

                for strike in K_call:
                    # countstrikecall+=1
                    # strike_put = torch.ones(1,1)*K_put[countstrikecall]
                    #  K_extended_put = torch.repeat_interleave(strike_put, MC_samples, dim=0).float()
                    # Since we use the same number of maturities for vanilla calls and puts: 
                    price = torch.clamp(S_old - strike, 0) - cv
                    var_price_no_cv = torch.var(torch.clamp(S_old - strike, 0))

                    # price_OTM = torch.cat([K_extended_put-S_old,zeros],1) #put OTM
                    # Discounting assumes we use 2-year time horizon 
                    var_price = torch.var(price)
                    # price_OTM = torch.max(price_OTM, 1, keepdim=True)[0]*torch.exp(-rate*1*i/n_steps)
                    Z_new = torch.cat([Z_new, price], 1)

                avg_S = Z_new.mean(dim=0, keepdim=True).T
                average_SS = torch.cat([average_SS, avg_S.T], 0)  # call OTM

        return average_SS, var_price, var_price_no_cv


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_models(seedused, iteration):
    loss_fn = nn.MSELoss()
    seedused = seedused + 1
    torch.manual_seed(seedused + 0 * 1000)
    np.random.seed(seedused + 0 * 1000)
    model = Net_SDE(dim=1, timegrid=timegrid, strikes_call=strikes_call, n_layers=2, vNetWidth=20, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, eps=1e-08, amsgrad=False, betas=(0.9, 0.999),
                                 weight_decay=0)
    lambda2 = lambda epoch: 0.985 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2)
    # optimizer= torch.optim.Rprop(model.parameters(), lr=0.001, etas=(0.5, 1.2), step_sizes=(1e-07, 1))
    n_epochs = 200
    itercount = 0
    loss_val_best = 10

    for epoch in range(n_epochs):
        MC_samples_gen = 200000
        n_steps = 360
        print('epoch:', epoch)

        batch_size = 20000
        permutation = torch.randperm(int(2 * MC_samples_gen))

        requires_grad_CV = (epoch % 2 != 0)
        requires_grad_SDE = not requires_grad_CV

        for p in model.diffusion.parameters():
            p.requires_grad = requires_grad_SDE
        for p in model.driftV.parameters():
            p.requires_grad = requires_grad_SDE
        for p in model.diffusionV.parameters():
            p.requires_grad = requires_grad_SDE
        for p in model.diffusionV1.parameters():
            p.requires_grad = requires_grad_SDE
        for p in model.control_variate.parameters():
            p.requires_grad = requires_grad_CV

        for i in range(0, 2 * MC_samples_gen, batch_size):
            indices2 = permutation[i:i + batch_size]
            batch_x = z_1[indices2, :]
            batch_y = z_2[indices2, :]
            timestart = time.time()
            optimizer.zero_grad()
            init_time = time.time()
            pred, var, var_price_no_cv = model(S0, V0, rate, BS_vol, indices, batch_x, batch_y, batch_size)
            time_forward = time.time() - init_time
            if requires_grad_CV:
                loss = var
                init_time = time.time()
                loss.backward()
                time_backward = time.time() - init_time
                print('iteration {}, variance={:.4f}, time_forward={:.4f}, time_backward={:.4f}'.format(itercount,
                                                                                                        loss.item(),
                                                                                                        time_forward,
                                                                                                        time_backward))
            else:
                loss = torch.sqrt(loss_fn(pred, target))
                losses.append(loss.clone().detach())
                itercount += 1
                init_time = time.time()
                loss.backward()
                time_backward = time.time() - init_time
                print('iteration {}, loss={:.4f}, time_forward={:.4f}, time_backward={:.4f}'.format(itercount,
                                                                                                    loss.item(),
                                                                                                    time_forward,
                                                                                                    time_backward))
            optimizer.step()
            print('time', time.time() - timestart)

        if requires_grad_SDE:
            scheduler.step()

        # evaluate and print RMSE validation error at the start of each epoch
        optimizer.zero_grad()
        with torch.no_grad():
            pred, _, _ = model(S0, V0, rate, BS_vol, indices, z_1, z_2, 2 * MC_samples_gen)

        loss_val = torch.sqrt(loss_fn(pred, target))
        print('validation {}, loss={:.4f}'.format(itercount, loss_val.item()))
        # scheduler.step()
        # print(get_lr(optimizer))

        if loss_val < loss_val_best:
            loss_val_best = loss_val
            print('loss_val_best', loss_val_best)
            path = "Neural_SDE_M0_BR_" + str(iteration) + ".pth"
            torch.save(model.state_dict(), path)

        # store the erorr value

        losses_val.append(loss_val.clone().detach())
    return seedused, model


# Load market prices and set training target
ITM_call = torch.load('ATM_target.pt').to(device=device)
target = ITM_call.float()

# Set up training
losses = []
losses_val = []
# strikes_put=[55,60, 65,70,75,80,85,90,95,100]
strikes_call = [100]  # ,105,110,115,120,125,130,135, 140,145]
seedsused = np.zeros((102, 1))
seedsused[0, 0] = -1
S0 = torch.ones(1, 1) * 100
BS_vol = torch.ones(1, 1) * 0.2
V0 = torch.ones(1, 1) * 0.04
rate = torch.ones(1, 1) * 0.025

n_steps = 360
# generate subdivisions of 2 year interval
timegrid = torch.linspace(0, 1, n_steps + 1)
# indices = torch.tensor([30,  60,  90, 120, 150, 180, 210, 240, 270, 300, 330, 360])
indices = torch.tensor([360])
# Start training 100 models
# fix the seeds for reproducibility
np.random.seed(1)
MC_samples_gen = 200000
z_1 = np.random.normal(size=(MC_samples_gen, n_steps))
z_2 = np.random.normal(size=(MC_samples_gen, n_steps))

# generate antithetics and pass to torch
z_1 = np.append(z_1, -z_1, axis=0)
z_2 = np.append(z_2, -z_2, axis=0)
z_1 = torch.tensor(z_1).to(device=device).float()
z_2 = torch.tensor(z_2).to(device=device).float()

for i in range(2, 103):
    np.save('seeds_used_M0.npy', seedsused)
    np.save('losses_M0.npy', losses)
    np.save('losses_val_M0.npy', losses_val)
    seedsused[i - 1, 0], model = train_models(int(seedsused[i - 2, 0]), i)
    path = "Neural_SDE_M0_" + str(i - 1) + ".pth"
    torch.save(model.state_dict(), path)
