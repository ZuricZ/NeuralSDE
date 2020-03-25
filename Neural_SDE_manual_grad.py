#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
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
        else:
            pass
        func(*args, **kwargs)
        if 'seed' in kwargs.keys():
            torch.manual_seed(torch.seed())
        else:
            pass
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
    def __init__(self, n_epochs=100, n_layers=2, vNetWidth=20, MC_samples=200000, batch_size0=1000, test_size=20000,
                 n_time_steps=48, S0=100, V0=0.04, rate=0.0, T=2):  # r = 0.025
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
            nn.init.normal_(m.weight, mean=0.0, std=0.05)  # for std > 0.1 paths quickly become singular
            try:
                m.bias.data.fill_(0.01)
            except:
                pass
        else:
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
    def __init__(self, dim, n_layers, vNetWidth):
        super(Net_SDE, self).__init__()

        self.dim = dim
        # Input to each coefficient (NN) will be (t,S_t,V_t)
        # We restrict diffusion coefficients to be positive
        self.diffusion = NeuralNetCell(dim=dim + 2, nOut=1, n_layers=n_layers, vNetWidth=vNetWidth, act_output='softp')
        self.diffusionV = NeuralNetCell(dim=dim + 1, nOut=1, n_layers=n_layers, vNetWidth=vNetWidth, act_output='softp')
        self.driftV = NeuralNetCell(dim=dim + 1, nOut=1, n_layers=n_layers, vNetWidth=vNetWidth)
        self.rho = NeuralNetCell(dim=1, nOut=1, n_layers=2, vNetWidth=5, act_output='tanh')  # restrict rho to [-1,1]


class SDE_tools(ModelParams):

    def __init__(self):
        ModelParams.__init__(self)
        # super(SDE_tools, self).__init__()

    @seed_decorator
    def generate_BMs(self, MC_samples, **kwargs):
        # create BM increments
        dW = torch.sqrt(self.h) * torch.randn((MC_samples, len(self.time_grid)))
        dB = torch.sqrt(self.h) * torch.randn((MC_samples, len(self.time_grid)))

        return dW, dB

    # @classmethod
    def generate_paths(self, model, W, scheme='absorption', detach_graph=False, no_grads=False):
        if scheme not in {'absorption', 'reflection'}:
            # absorption ensures positive volatility
            raise ValueError('Wrong discretization scheme.')
        else:
            pass

        # create BM increments
        dW, dB = W[0], W[1]

        MC_samples = dW.shape[0]
        zeros, ones = torch.zeros(MC_samples, 1), torch.ones(MC_samples, 1)
        S, V = torch.zeros((MC_samples, len(self.time_grid))), torch.zeros((MC_samples, len(self.time_grid)))
        S[:, 0], V[:, 0] = self.S0, self.V0

        with torch.no_grad() if no_grads else ExitStack() as gs:
            # Euler-Maruyama S_t, V_t
            for idx, t in enumerate(self.time_grid[1:]):
                input_S = torch.cat([ones * t, S[:, idx, None], V[:, idx, None]], 1)
                input_V = torch.cat([ones * t, V[:, idx, None]], 1)

                # absorption ensures positive stock price
                S[:, idx+1] = torch.max(S[:, idx] * (1 + self.rate * self.h) +
                                        model.diffusion(input_S).squeeze() * dW[:, idx], zeros.squeeze())

                V_temp = V[:, idx, None] + model.driftV(input_V) * self.h + model.diffusionV(input_V) * (
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

    def calc_prices(self, S, detach_graph=False):
        price_call_mat = torch.zeros(len(self.maturities), len(self.strikes))
        price_put_mat = torch.zeros(len(self.maturities), len(self.strikes))
        discount_factor = lambda t: torch.exp(-self.rate * t)
        S_T = S[:, self.maturities]
        for idx, strike in enumerate(self.strikes):
            call_price_vec = (torch.max(S_T - strike, torch.zeros_like(S_T)) * discount_factor(
                self.time_grid[self.maturities])).mean(dim=0)
            put_price_vec = (torch.max(strike - S_T, torch.zeros_like(S_T)) * discount_factor(
                self.time_grid[self.maturities])).mean(dim=0)
            price_call_mat[:, idx], price_put_mat[:, idx] = call_price_vec, put_price_vec

        if detach_graph:
            return torch.cat([price_call_mat, price_put_mat], 0).detach()
        else:
            return torch.cat([price_call_mat, price_put_mat], 0)


class Trainer(SDE_tools):
    def __init__(self, model_train, learning_rate=0.1, milestones=None, gamma=0.1):
        SDE_tools.__init__(self)
        # super(Trainer, self).__init__()
        if milestones is None:
            milestones = [100, 200]

        self.model_train = model_train
        self.optimizer = torch.optim.Adam(model_train.parameters(), lr=learning_rate, eps=1e-08, amsgrad=True,
                                          betas=(0.9, 0.999), weight_decay=0)
        # optimizer= torch.optim.Rprop(model.parameters(), lr=0.001, etas=(0.5, 1.2), step_sizes=(1e-07, 1))
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)
        self.scheduler_func = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.985 ** epoch)

    def loss_func(self, C, C_mkt):
        return torch.mean(torch.pow(C - C_mkt, 2))

    def loss_SPX_VIX(self, C, C_mkt, Fwd, Fwd_mkt):
        # weighted loss of VIX Fwds and SPX options
        N, M = len(self.maturities), len(self.strikes)
        return ((N + M)*self.loss_func(C, C_mkt) + N*self.loss_func(Fwd, Fwd_mkt))/(2*N + M)

    @staticmethod
    @log_time
    def torch_backprop(loss):
        loss.backward()

    def train_models(self, C_mkt, P_mkt):
        # Batch size as a function of upper bound on MC error
        def batch_func(loss):
            if loss:
                return int(np.minimum(self.batch_size0, 2.576**2/(4*0.01**2*np.power(loss, 2)))) + 1000
            else:
                return self.batch_size0

        true_prices = torch.cat([C_mkt, P_mkt], 0)
        W_test = self.generate_BMs(self.test_size, seed=0)

        plot_grads = utils.GradPlot()

        for epoch in range(self.n_epochs):
            # evaluate and print test error at the start of each epoch
            S_test, V_test = self.generate_paths(self.model_train, W_test, no_grads=True, detach_graph=True)

            test_prices = self.calc_prices(S_test, detach_graph=True)
            loss_test = self.loss_func(test_prices, true_prices)

            batch_size = batch_func(False)
            print('--------------------------')
            print('Epoch: {} ~ Batch size: {}'.format(epoch, batch_size))
            print('\t Test Loss={0:.4f}'.format(torch.sqrt(loss_test).item()))

            # generate paths TODO: generate paths in the main()?
            BMs = self.generate_BMs(batch_size, seed=epoch)
            S, V = self.generate_paths(self.model_train, BMs, no_grads=True, detach_graph=True)  # detach from graph/no graphs if running manual backprop else not
            # prices = self.calc_prices(S)
            # loss = self.loss_func(prices, true_prices)

            self.optimizer.zero_grad()
            # self.torch_backprop(loss)
            # self.optimizer.step()
            # self.scheduler_func.step()
            # print('\t Train Loss={0:.4f}'.format(torch.sqrt(loss).item()))  # previous epoch loss in reality
            # plot_grads.replot_training(self.model_train.diffusion.h_h[0][0].weight.grad)

            backward = backprop(self.model_train, S, V, BMs)
            grad_dict = backward.loss_grad(C_mkt, P_mkt)
            Optimizers.SGD(self.model_train, grad_dict)

            plot_grads.replot_training(grad_dict['h_h.0.0.weight'])
            # utils.GradPlot.plt_grads(grad_dict['h_h.0.0.weight'], f'grad in loop {epoch}')

        # return self.model_train


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

    train_time = time.time()
    model = Net_SDE(dim=1, n_layers=params.n_layers, vNetWidth=params.vNetWidth)  # Model will be actually changed inplace
    # utils.GradPlot.plt_grads(model.diffusion.h_h[0][0].weight, 'weight')
    tools = SDE_tools()
    # print(f'Out: {id(model)}')
    trainer = Trainer(model)
    trainer.train_models(C_mkt, P_mkt)

    path = "Neural_SDE" + ".pth"
    torch.save(model.state_dict(), path)
    print('--- MODEL TRAINING TIME: %d min %d s ---' % divmod(time.time() - train_time, 60))


print('done')
