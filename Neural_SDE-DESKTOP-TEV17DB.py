#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import numpy as np
import math
import os
import time
import itertools

if torch.cuda.is_available():
    device = 'cuda'
    # Uncomment below to pick particular device if running on a cluster:
    # torch.cuda.set_device(6) 
    # device='cuda:6'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.cuda.current_device()

else:
    device = 'cpu'
    torch.set_default_tensor_type('torch.FloatTensor')


class Net_timestep(nn.Module):
    """
    Set up NN-class
    Name Net_timesteps corresponds to using
    different NNs for each timestep but in the current set up
    coefficients of SDE are neural networks
    and time is one of the inputs
    """

    def __init__(self, dim, nOut, n_layers, vNetWidth, activation="relu", diffusion=False):
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
        if not diffusion:
            self.h_o = self.outputLayer(vNetWidth, nOut)
        else:
            self.relu_activation = nn.ReLU()
            self.h_o = self.outputLayer(vNetWidth, nOut, relu=True)

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

    def outputLayer(self, nIn, nOut, relu=False):
        if not relu:
            layer = nn.Sequential(nn.Linear(nIn, nOut))  # , bias=True
            return layer
        else:
            layer = nn.Sequential(nn.Linear(nIn, nOut), self.relu_activation)  # , bias=True
            return layer

    def forward(self, S):
        h = self.i_h(S)
        for l in range(len(list(self.h_h))):
            h = list(self.h_h)[l](h)
        output = self.h_o(h)
        return output


# Set up neural SDE class
class Net_SDE(nn.Module):

    def __init__(self, dim, timegrid, strikes_call, strikes_put, ITM_call, OTM_call, ITM_put, OTM_put, n_layers,
                 vNetWidth, device, rho=0):

        super(Net_SDE, self).__init__()
        self.dim = dim
        self.timegrid = timegrid
        self.device = device
        self.strikes_call = strikes_call
        self.strikes_put = strikes_put
        self.rho = rho

        # Input to each coefficient (NN) will be (t,S_t,V_t)
        # We restrict diffusion coefficients to be positive
        self.diffusion = Net_timestep(dim=dim + 2, nOut=1, n_layers=n_layers, vNetWidth=vNetWidth, diffusion=True)
        self.driftV = Net_timestep(dim=dim + 2, nOut=1, n_layers=n_layers, vNetWidth=vNetWidth)
        self.diffusionV = Net_timestep(dim=dim + 2, nOut=1, n_layers=n_layers, vNetWidth=vNetWidth, diffusion=True)
        self.diffusionV1 = Net_timestep(dim=dim + 2, nOut=1, n_layers=n_layers, vNetWidth=vNetWidth, diffusion=True)

    def forward(self, S0, V0, rate, indices, z, z1):
        MC_samples = z.shape[0]
        zeros = torch.zeros(MC_samples, 1)
        ones = torch.ones(MC_samples, 1)
        S_old, V_old = ones * S0, ones * V0
        K_call, K_put = self.strikes_call, self.strikes_put

        call_payoff = lambda x, K: torch.max(x - K, zeros)
        put_payoff = lambda x, K: torch.max(K - x, zeros)

        price_call_OTM_mat, price_put_OTM_mat = torch.Tensor(), torch.Tensor()  # price matrix call OTM
        price_call_ITM_mat, price_put_ITM_mat = torch.Tensor(), torch.Tensor()  # price matrix call ITM

        # use fixed step size
        h = self.timegrid[1] - self.timegrid[0]
        n_steps = len(self.timegrid) - 1

        # Solve for S_t, V_t (Euler)
        for i in range(1, len(self.timegrid)):
            dW = (torch.sqrt(h) * z[:, i - 1]).reshape(MC_samples, 1)
            dW1 = (torch.sqrt(h) * z1[:, i - 1]).reshape(MC_samples, 1)
            input_time = ones * self.timegrid[i - 1]
            inputNN = torch.cat([input_time, S_old, V_old], 1)

            S_new = S_old + S_old * rate * h + self.diffusion(inputNN) * dW
            S_old = torch.max(S_new, zeros)  # absorption ensures positive stock price

            V_new = V_old + self.driftV(inputNN) * h + self.diffusionV(inputNN) * dW + self.diffusionV1(inputNN) * dW1  # TODO: rewrite this
            V_old = torch.max(V_new, zeros)  # absorption ensures positive volatility

            discount_factor = torch.exp(-rate * 2 * i / n_steps)

            # If particular timestep is a maturity for Vanilla option
            if i in indices:
                price_call_OTM_vec, price_put_OTM_vec = torch.zeros(len(K_call), 1), torch.zeros(len(K_put), 1)
                price_call_ITM_vec, price_put_ITM_vec = torch.zeros(len(K_put), 1), torch.zeros(len(K_call), 1)

                # Evaluate put (OTM/ITM) and call (OTM/ITM) option prices
                for idx, (strike, strike_put) in enumerate(itertools.zip_longest(K_call, K_put, fillvalue=np.nan)):
                    strike_tensor = torch.ones(MC_samples, 1) * strike
                    strike_put_tensor = torch.ones(MC_samples, 1) * strike_put

                    # avoid creating new variables (RAM usage)
                    price_call_OTM_vec[idx], price_put_OTM_vec[idx], \
                    price_call_ITM_vec[idx], price_put_ITM_vec[idx] = \
                        (call_payoff(S_old, strike_tensor) * discount_factor).mean(), \
                        (put_payoff(S_old, strike_put_tensor) * discount_factor).mean(), \
                        (call_payoff(S_old, strike_put_tensor) * discount_factor).mean(), \
                        (put_payoff(S_old, strike_tensor) * discount_factor).mean()

                price_call_OTM_mat = torch.cat([price_call_OTM_mat, price_call_OTM_vec.T], 0)  # call OTM
                price_put_OTM_mat = torch.cat([price_put_OTM_mat, price_put_OTM_vec.T], 0)  # put OTM

                price_call_ITM_mat = torch.cat([price_call_ITM_mat, price_call_ITM_vec.T], 0)  # call ITM
                price_put_ITM_mat = torch.cat([price_put_ITM_mat, price_put_ITM_vec.T], 0)  # put ITM

        # Return model implied vanilla option prices
        return torch.cat([price_call_OTM_mat, price_put_OTM_mat, price_call_ITM_mat, price_put_ITM_mat], 0)


# def price_options(i, S_old, strike_tensor, strike_put_tensor): # TODO: adapt to handle NaN (different lens of strikes)
#     zeros = torch.zeros_like(S_old)
#     # Since we use the same number of maturities for vanilla calls and puts:
#     payoff_call_OTM = torch.max(S_old - strike_tensor, zeros)  # call OTM
#     payoff_put_OTM = torch.max(strike_put_tensor - S_old, zeros)  # put OTM
#
#     payoff_call_ITM = torch.max(S_old - strike_put_tensor, zeros)  # call ITM
#     payoff_put_ITM = torch.max(strike_tensor - S_old, zeros)  # put ITM
#
#     # Discounting assumes we use 2-year time horizon
#     discounted_payoff_call_OTM = payoff_call_OTM * torch.exp(-rate * 2 * i / n_steps)
#     discounted_payoff_put_OTM = payoff_put_OTM * torch.exp(-rate * 2 * i / n_steps)
#
#     discounted_payoff_call_ITM = payoff_call_ITM * torch.exp(-rate * 2 * i / n_steps)
#     discounted_payoff_put_ITM = payoff_put_ITM * torch.exp(-rate * 2 * i / n_steps)
#
#     # Calculate price via MC
#     price_call_OTM = discounted_payoff_call_OTM.mean()
#     price_put_OTM = discounted_payoff_put_OTM.mean()
#
#     price_call_ITM = discounted_payoff_call_ITM.mean()
#     price_put_ITM = discounted_payoff_put_ITM.mean()
#     return price_call_OTM, price_put_OTM, price_call_ITM, price_put_ITM


def train_models(seedused):
    loss_fn = nn.MSELoss()
    seedused = seedused + 1
    torch.manual_seed(seedused)
    np.random.seed(seedused)
    model = Net_SDE(dim=1, timegrid=timegrid, strikes_call=strikes_call, strikes_put=strikes_put, ITM_call=ITM_call,
                    OTM_call=OTM_call, ITM_put=ITM_put, OTM_put=OTM_put, n_layers=2, vNetWidth=20, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-08, amsgrad=False, betas=(0.9, 0.999),
                                 weight_decay=0)
    # optimizer= torch.optim.Rprop(model.parameters(), lr=0.001, etas=(0.5, 1.2), step_sizes=(1e-07, 1))
    n_epochs = 15  # 200
    itercount = 0

    MC_samples_gen = 200000
    n_steps = 48
    batch_size0 = 20000
    # batch_func = lambda x: 5000 + int(batch_size*np.sqrt(x / n_epochs))
    batch_func = lambda x: 5000 + int(batch_size0 / 2 * np.sqrt(x / n_epochs)) + (batch_size0 // 2 if x > 0 else 0)

    # fix the seeds for reproducibility
    np.random.seed(0 + seedused * 1000)
    z_1 = np.random.normal(size=(MC_samples_gen, n_steps))
    np.random.seed(1 + seedused * 1000)
    z_2 = np.random.normal(size=(MC_samples_gen, n_steps))

    # generate antithetics and pass to torch
    z_1 = np.append(z_1, -z_1, axis=0)
    z_2 = np.append(z_2, -z_2, axis=0)
    z_1 = torch.tensor(z_1).to(device=device).float()
    z_2 = torch.tensor(z_2).to(device=device).float()

    np.random.seed(2 + seedused * 1000)
    z_1_test = np.random.normal(size=(MC_samples_gen // 4, n_steps))  # TODO: fix to torch
    np.random.seed(3 + seedused * 1000)
    z_2_test = np.random.normal(size=(MC_samples_gen // 4, n_steps))

    for epoch in range(n_epochs):
        print('--------------------------')
        print('Epoch: {} ~ Batch size: {}'.format(epoch, batch_func(epoch)))

        # evaluate and print RMSE validation error at the start of each epoch
        optimizer.zero_grad()
        pred = model(S0, V0, rate, indices, z_1_test, z_2_test).detach()
        loss_val = torch.sqrt(loss_fn(pred, target))
        print('Test {0}, loss={1:.4f}'.format(itercount, loss_val.item()))

        # store the error value
        losses_val.append(loss_val.clone().detach())

        # TODO: consider using the same MC samples each epoch?
        # randomly reshufle samples and then use subsamples for training
        # this is useful when we want to reuse samples for each epoch
        permutation = torch.randperm(int(2 * MC_samples_gen))
        batch_size = batch_func(epoch)

        for i in range(0, 2 * MC_samples_gen, batch_size):
            indices2 = permutation[i: i + batch_size]
            batch_x, batch_y = z_1[indices2, :], z_2[indices2, :]
            timestart = time.time()

            optimizer.zero_grad()
            pred = model(S0, V0, rate, indices, batch_x, batch_y)
            loss = torch.sqrt(loss_fn(pred, target))

            loss.backward()
            optimizer.step()

            losses.append(loss.clone().detach())
            itercount += 1

            print('iteration {0}, loss={1:.4f}'.format(itercount, loss.item()))
            print('time: {0:.2f}s'.format(time.time() - timestart))

    return seedused, model


# Load market prices and set training target
ITM_call = torch.load('ITM_call.pt').to(device=device)
ITM_put = torch.load('ITM_put.pt').to(device=device)
OTM_call = torch.load('OTM_call.pt').to(device=device)
OTM_put = torch.load('OTM_put.pt').to(device=device)
target = torch.cat([OTM_call, OTM_put, ITM_call, ITM_put], 0)

# Set up training
losses, losses_val = [], []
strikes_put = [55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
strikes_call = [100, 105, 110, 115, 120, 125, 130, 135, 140, 145]
seedsused = np.zeros((101, 1))
seedsused[0, 0] = -1

S0 = torch.ones(1, 1) * 100
V0 = torch.ones(1, 1) * 0.04
rate = torch.ones(1, 1) * 0.025

n_steps = 48
# generate subdivisions of 2 year interval
timegrid = torch.linspace(0, 2, n_steps + 1)
# If using n_steps=48 those corresponds to monthly maturities:
indices = torch.tensor([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,
                        38, 40, 42, 44, 46, 48])

# Start training 100 models
for i in range(2, 102):
    np.save('seeds_used.npy', seedsused)
    np.save('losses.npy', losses)
    np.save('losses_val.npy', losses_val)
    seedsused[i - 1, 0], model = train_models(int(seedsused[i - 2, 0]))
    path = "Neural_SDE_" + str(i - 1) + ".pth"
    torch.save(model.state_dict(), path)
    break


def MC_paths(model, S0, V0, rate, timegrid, N=1000):
    model.eval()
    h = timegrid[1] - timegrid[0]
    S_old, V_old = torch.repeat_interleave(S0, N, dim=0), torch.repeat_interleave(V0, N, dim=0)
    S_path, V_path = np.zeros((N, len(timegrid))), np.zeros((N, len(timegrid)))
    S_path[:, 0], V_path[:, 0] = S0, V0

    for idx, t in enumerate(timegrid[1:]):
        dW, dW1 = torch.sqrt(h) * torch.randn((N, 1)), torch.sqrt(h) * torch.randn((N, 1))
        input_time = torch.ones(N, 1) * t
        inputNN = torch.cat([input_time, S_old.float(), V_old.float()], 1)

        S_new = S_old + S_old * rate * h + model.diffusion(inputNN) * dW

        V_new = torch.max(V_old + model.driftV(inputNN) * h
                          + model.diffusionV(inputNN) * dW
                          + model.diffusionV1(inputNN) * dW1, torch.ones_like(V_old))

        S_path[:, idx + 1] = np.squeeze(S_new.detach().numpy())
        S_old = S_new

        V_path[:, idx + 1] = np.squeeze(V_new.detach().numpy())
        V_old = V_new

    return S_path, V_path


print('done')
