#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torchviz
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
# from backward_pass import backprop
# from backward_pass import Optimizers
import utils  # mostly plotting functions
import time
import matplotlib.pyplot as plt
from contextlib import ExitStack
import itertools
import math
import os

torch.autograd.set_detect_anomaly(True)


def seed_decorator(func):
    def set_seed(*args, **kwargs):
        if 'seed' in kwargs.keys():
            torch.manual_seed(kwargs['seed'])

        func(*args, **kwargs)

        if 'seed' in kwargs.keys():
            torch.manual_seed(torch.seed())
        return func(*args, **kwargs)
    return set_seed


def log_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        retval = func(*args, **kwargs)
        end_time = time.time()
        print("\t Elapsed time: %.2f" % (end_time - start_time))
        return retval
    return wrapper


class ModelParams:
    def __init__(self, n_epochs=100, n_layers=2, vNetWidth=20, MC_samples=20000, batch_size0=10000, test_size=20000,
                 n_time_steps=48, S0=100, V0=0.04, rate=0.025, T=2):  # r = 0.025
        self.n_epochs = n_epochs
        self.n_layers = n_layers
        self.vNetWidth = vNetWidth
        self.MC_samples = MC_samples
        self.n_time_steps = n_time_steps
        self.batch_size0 = batch_size0  # maximum batch size
        self.test_size = test_size
        self.T = T
        self.time_grid = torch.linspace(0, self.T, self.n_time_steps + 1)
        self.h = self.time_grid[1] - self.time_grid[0]  # use fixed step size
        self.strikes_put = [55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
        self.strikes_call = [100, 105, 110, 115, 120, 125, 130, 135, 140, 145]
        self.strikes = self.strikes_put + self.strikes_call
        monthly_step = n_time_steps // 24  # we have 24 maturities worth of Heston option prices
        self.maturities = [int(maturity) for maturity in range(monthly_step, n_time_steps + monthly_step, monthly_step)]
        self.S0 = S0
        self.V0 = V0
        self.rate = rate


class NeuralNetCell(nn.Module):
    def __init__(self, dim, nOut, n_layers, vNetWidth, activation="relu", act_output='none'):
        super(NeuralNetCell, self).__init__()
        self.dim = dim
        self.nOut = nOut
        self.relu_activation = nn.ReLU()
        self.tanh_activation = nn.Tanh()
        self.elu_activation = nn.ELU()
        self.softp_activation = nn.Softplus()

        if activation not in {'relu', 'tanh'}:
            raise ValueError("unknown activation function {}".format(activation))
        elif activation == "relu":
            self.activation = self.relu_activation
        else:
            self.activation = self.tanh_activation

        self.i_h = self.hiddenLayerT1(dim, vNetWidth)
        self.h_h = nn.ModuleList([self.hiddenLayerT1(vNetWidth, vNetWidth) for l in range(n_layers - 1)])
        self.h_o = self.outputLayer(vNetWidth, nOut, act=act_output)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            # nn.init.xavier_normal_(m.weight, gain=0.5)
            nn.init.normal_(m.weight, mean=0.0, std=0.1)  # for std > 0.1 paths quickly become singular
            try:
                m.bias.data.fill_(0.01)
            except:
                pass

    def hiddenLayerT0(self, nIn, nOut):
        layer = nn.Sequential(  # nn.BatchNorm1d(nIn, momentum=0.1),
            nn.Linear(nIn, nOut, bias=True),
            # nn.BatchNorm1d(nOut, momentum=0.1),
            self.activation)
        layer.apply(self.init_weights)
        return layer

    def hiddenLayerT1(self, nIn, nOut):
        layer = nn.Sequential(nn.Linear(nIn, nOut, bias=True),
                              # nn.BatchNorm1d(nOut, momentum=0.1),
                              self.activation)
        layer.apply(self.init_weights)
        return layer

    def outputLayer(self, nIn, nOut, act='none'):
        if act == 'none':
            layer = nn.Sequential(nn.Linear(nIn, nOut, bias=False))
        elif act == 'relu':
            layer = nn.Sequential(nn.Linear(nIn, nOut, bias=False), self.relu_activation)
        elif act == 'elu':
            layer = nn.Sequential(nn.Linear(nIn, nOut, bias=False), self.elu_activation)
        elif act == 'softp':
            layer = nn.Sequential(nn.Linear(nIn, nOut, bias=False), self.softp_activation)
        elif act == 'tanh':
            layer = nn.Sequential(nn.Linear(nIn, nOut, bias=False), self.tanh_activation)
        else:
            raise ValueError('Wrong activation.')
        layer.apply(self.init_weights)
        return layer

    def forward(self, S):
        h = self.i_h(S)
        for l in range(len(list(self.h_h))):
            h = list(self.h_h)[l](h)
        output = self.h_o(h)
        return output


# Set up neural SDE class
class Net_SDE(nn.Module):
    def __init__(self, params):
        super(Net_SDE, self).__init__()
        self.params = params
        self.tools = SDE_tools(params)
        # Input to coefficient (NN) will be (t,S_t,V_t)
        self.diffusion = NeuralNetCell(dim=3, nOut=1, n_layers=params.n_layers, vNetWidth=params.vNetWidth,
                                       act_output='softp')  # We restrict diffusion coefficients to be positive
        # Input to each coefficient (NN) will be (t,V_t)
        self.diffusionV = NeuralNetCell(dim=2, nOut=1, n_layers=params.n_layers, vNetWidth=params.vNetWidth,
                                        act_output='softp')  # We restrict diffusion coefficients to be positive
        self.driftV = NeuralNetCell(dim=2, nOut=1, n_layers=params.n_layers, vNetWidth=params.vNetWidth)
        # Input to each coefficient (NN) will be (t)
        self.rho = NeuralNetCell(dim=1, nOut=1, n_layers=2, vNetWidth=5,
                                 act_output='tanh')  # restrict rho to [-1,1]

    def forward(self, W):
        dW, dB = W
        zeros, ones = torch.zeros(dW.shape[0], 1), torch.ones(dW.shape[0], 1)

        S_t, V_t = ones * self.params.S0, ones * self.params.V0

        price_call_mat = torch.zeros(len(self.params.maturities), len(self.params.strikes))
        price_put_mat = torch.zeros(len(self.params.maturities), len(self.params.strikes))

        discount_factor = lambda t: torch.exp(-self.params.rate * t)

        # Euler-Maruyama S_t, V_t
        for idx, t in enumerate(self.params.time_grid[:-1], 1):
            # S_t, V_t = self.tools.generate_path_step(self, (dW[:, idx], dB[:, idx]), (S_t, V_t), t)

            input_S = torch.cat([ones * t, S_t, V_t], 1)
            input_V = input_S[:, [0, 2]]

            # absorption ensures positive stock price
            S_new = S_t * (1 + self.params.rate * self.params.h) + self.diffusion(input_S) * dW[:, idx-1, None]
            S_t = torch.max(S_new, zeros)

            V_new = V_t + self.driftV(input_V) * self.params.h + self.diffusionV(input_V) * (
                    torch.sqrt(1 - torch.pow(self.rho(ones * t), 2)) * dB[:, idx-1, None]
                    + self.rho(ones * t) * dW[:, idx-1, None])
            V_t = torch.max(V_new, zeros)

            if idx in self.params.maturities:  # TODO: check how many times idx is called
                # price_call_mat[idx, :], price_put_mat[idx, :] = self.tools.calc_prices(S_t, mat=t)
                idx_t = self.params.maturities.index(idx)
                for idx_k, strike in enumerate(self.params.strikes):
                    price_call_mat[idx_t, idx_k] = (torch.max(S_t - strike, torch.zeros_like(S_t)) *
                                              discount_factor(t)).mean()
                    price_put_mat[idx_t, idx_k] = (torch.max(strike - S_t, torch.zeros_like(S_t)) *
                                             discount_factor(t)).mean()

        return torch.cat([price_call_mat, price_put_mat], 0)


class SDE_tools:

    def __init__(self, params):
        self.params = params
        # ModelParams.__init__(self)

    @staticmethod
    def normalize_input(tensor):
        return tensor - tensor.mean(0)

    @seed_decorator
    def generate_BMs(self, MC_samples, **kwargs):
        # create BM increments
        dW = torch.sqrt(self.params.h) * torch.randn((MC_samples, len(self.params.time_grid)))
        dB = torch.sqrt(self.params.h) * torch.randn((MC_samples, len(self.params.time_grid)))

        return dW, dB

    def generate_path_step(self, model, W_t, X_t, t, scheme='absorption'):
        if scheme not in {'absorption', 'reflection'}:
            raise ValueError('Wrong discretization scheme.')

        dW_t, dB_t = W_t[0].view(-1, 1), W_t[1].view(-1, 1)
        S_old, V_old = X_t[0].view(-1, 1), X_t[1].view(-1, 1)

        zeros, ones = torch.zeros(len(dW_t), 1), torch.ones(len(dW_t), 1)

        input_S = torch.cat([ones * t, S_old, V_old], 1)
        input_V = input_S[:, [0, 2]]

        # absorption ensures positive stock price
        S_new = torch.max(S_old * (1 + self.params.rate * self.params.h) + model.diffusion(input_S) * dW_t, zeros)

        V_temp = V_old + model.driftV(input_V) * self.params.h + model.diffusionV(input_V) * (
                torch.sqrt(1 - torch.pow(model.rho(ones * t), 2)) * dB_t
                + model.rho(ones * t) * dW_t)
        V_new = torch.max(V_temp, zeros) if scheme == 'absorption' else torch.abs(V_temp)

        return S_new, V_new

    def generate_paths(self, model, W, scheme='absorption', detach_graph=False, no_grads=False):
        if scheme not in {'absorption', 'reflection'}:
            # absorption ensures positive volatility
            raise ValueError('Wrong discretization scheme.')

        # create BM increments
        dW, dB = W[0], W[1]

        MC_samples = dW.shape[0]
        zeros, ones = torch.zeros(MC_samples, 1), torch.ones(MC_samples, 1)
        S, V = torch.zeros((MC_samples, len(self.params.time_grid))), torch.zeros((MC_samples, len(self.params.time_grid)))
        S[:, 0], V[:, 0] = self.params.S0, self.params.V0

        with torch.no_grad() if no_grads else ExitStack() as gs:
            # Euler-Maruyama S_t, V_t
            for idx, t in enumerate(self.params.time_grid[1:]):
                input_S = torch.cat([ones * t, S[:, idx, None], V[:, idx, None]], 1)
                input_V = torch.cat([ones * t, V[:, idx, None]], 1)

                # De-mean the input
                # input_S = self.normalize_input(input_S)
                # input_V = self.normalize_input(input_V)

                # absorption ensures positive stock price
                S[:, idx+1] = torch.max(S[:, idx] * (1 + self.params.rate * self.params.h) +
                                        model.diffusion(input_S).squeeze() * dW[:, idx], zeros.squeeze())

                V_temp = V[:, idx, None] + model.driftV(input_V) * self.params.h + model.diffusionV(input_V) * (
                        torch.sqrt(1 - torch.pow(model.rho(ones * t), 2)) * dB[:, idx, None]
                        + model.rho(ones * t) * dW[:, idx, None])
                V[:, idx+1] = torch.max(V_temp.squeeze(), zeros.squeeze()) \
                    if scheme == 'absorption' else torch.abs(V_temp)

                # print(self.driftV(input_V))
                # print(self.diffusionV(input_V))
                # print(self.rho(ones * t))

        if detach_graph:
            return S.detach(), V.detach()
        else:
            return S, V

    def generate_VIX(self, V):
        pass

    def calc_prices(self, S, mat=None, no_grads=False):
        discount_factor = lambda t: torch.exp(-self.params.rate * t)
        condition = S.squeeze().ndimension() <= 1
        if condition:
            price_call_mat = torch.zeros(1, len(self.params.strikes))
            price_put_mat = torch.zeros(1, len(self.params.strikes))
            S_T = S.view(-1, 1)
        else:
            price_call_mat = torch.zeros(len(self.params.maturities), len(self.params.strikes))
            price_put_mat = torch.zeros(len(self.params.maturities), len(self.params.strikes))
            S_T = S[:, self.params.maturities]
            mat = self.params.time_grid[self.params.maturities]

        with torch.no_grad() if no_grads else ExitStack() as gs:
            for idx, strike in enumerate(self.params.strikes):
                price_call_mat[:, idx] = (torch.max(S_T - strike, torch.zeros_like(S_T)) *
                                          discount_factor(mat)).mean(dim=0)
                price_put_mat[:, idx] = (torch.max(strike - S_T, torch.zeros_like(S_T)) *
                                         discount_factor(mat)).mean(dim=0)

        if condition:
            return price_call_mat.squeeze(), price_put_mat.squeeze()
        else:
            return torch.cat([price_call_mat, price_put_mat], 0)


class Trainer:
    def __init__(self, params, learning_rate=0.05, milestones=None, gamma=0.1):
        # super(Trainer, self).__init__()
        if milestones is None:
            milestones = [100, 200]

        self.params = params
        self.tools = SDE_tools(params)
        self.learning_rate = learning_rate

    def loss_func(self, C, C_mkt):
        return torch.mean(torch.pow(C - C_mkt, 2))

    def loss_SPX_VIX(self, C, C_mkt, Fwd, Fwd_mkt):
        # weighted loss of VIX Fwds and SPX options
        N, M = len(self.params.maturities), len(self.params.strikes)
        return ((N + M)*self.loss_func(C, C_mkt) + N*self.loss_func(Fwd, Fwd_mkt))/(2*N + M)

    @staticmethod
    @log_time
    def torch_backprop(loss):
        loss.backward()

    @staticmethod
    @log_time
    def torch_forward(model_train, BMs):
        return model_train(BMs)

    def batch_func(self, loss):
        # Batch size as a function of upper bound on MC error
        if loss:
            return int(np.minimum(self.params.batch_size0, 2.576 ** 2 / (4 * 0.01 ** 2 * np.power(loss, 2)))) + 1000
        else:
            return self.params.batch_size0

    def train_models(self, true_prices):
        W_test = self.tools.generate_BMs(self.params.test_size, seed=0)

        model = Net_SDE(params)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, eps=1e-08, amsgrad=True,
                                          betas=(0.9, 0.999), weight_decay=0)
        # optimizer= torch.optim.Rprop(model.parameters(), lr=0.001, etas=(0.5, 1.2), step_sizes=(1e-07, 1))
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)
        scheduler_func = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.985 ** epoch)

        # plot_grads = utils.GradPlot()

        for epoch in range(self.params.n_epochs):
            print('--------------------------')
            print('Epoch: {}'.format(epoch))
            # evaluate and print test error at the start of each epoch
            # S_test, V_test = self.tools.generate_paths(model, W_test, no_grads=True, detach_graph=True)
            # test_prices = self.tools.calc_prices(S_test, no_grads=True)

            print('Test:\t\t', end='')
            with torch.no_grad():
                test_prices = self.torch_forward(model, W_test)
                loss_test = self.loss_func(test_prices, true_prices).item()
            # plt.figure(2)
            # plt.plot(S_test[:100, :].T)
            # plt.show()

            batch_size = self.batch_func(False)

            print('\t Test Loss = *{0:.4f}* ~ Batch size for train: {1}'.format(loss_test, batch_size))

            # generate paths TODO: generate paths in the main()?
            BMs = self.tools.generate_BMs(batch_size, seed=epoch)

            # forward pass
            print('Forward pass: ', end='')
            model_prices = self.torch_forward(model, BMs)  # time it
            loss = self.loss_func(model_prices, true_prices)

            # backward pass
            print('Backward pass: ', end='')
            optimizer.zero_grad()
            self.torch_backprop(loss)  # time it

            optimizer.step()
            scheduler_func.step()

            # print('\t Train Loss={0:.4f}'.format(torch.sqrt(loss).item()))  # previous epoch loss in reality
            # plot_grads.replot_training(self.model_train.diffusion.h_h[0][0].weight.grad)

            # plot_grads.replot_training(grad_dict['h_h.0.0.weight'])
            # utils.GradPlot.plt_grads(grad_dict['h_h.0.0.weight'], f'grad in loop {epoch}')

        return model


if __name__ == "__main__":
    torch.manual_seed(1)
    params = ModelParams()
    timegrid = params.time_grid
    # Load market prices and set training target
    ITM_call = torch.load('ITM_call.pt')
    ITM_put = torch.load('ITM_put.pt')
    OTM_call = torch.load('OTM_call.pt')
    OTM_put = torch.load('OTM_put.pt')
    C_mkt = torch.cat([ITM_call, OTM_call], 1)[:len(params.maturities), :]
    P_mkt = torch.cat([OTM_put, ITM_put], 1)[:len(params.maturities), :]

    # writer = SummaryWriter()
    train_time = time.time()
    # writer.add_graph(model.diffusion)
    # utils.GradPlot.plt_grads(model.diffusion.h_h[0][0].weight, 'weight')
    tools = SDE_tools(params)
    trainer = Trainer(params)
    model = trainer.train_models(torch.cat([C_mkt, P_mkt], 0))

    path = "Neural_SDE" + ".pth"
    torch.save(model.state_dict(), path)
    print('--- MODEL TRAINING TIME: %d min %d s ---' % divmod(time.time() - train_time, 60))


print('done')
