#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import numpy as np
import utils  # mostly plotting functions
import time
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from contextlib import ExitStack
import itertools
import math
import os
from heston_VIX import heston_VIX2
import calc_implied_vol


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


def seed_decorator(func):
    def set_seed(*args, **kwargs):
        if 'seed' in kwargs.keys():
            torch.manual_seed(kwargs['seed'])

        func(*args, **kwargs)

        if 'seed' in kwargs.keys():
            # torch.manual_seed(torch.seed())
            torch.default_generator.seed()
        return func(*args, **kwargs)

    return set_seed


def log_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        retval = func(*args, **kwargs)
        end_time = time.time()
        print('Elapsed time: {:.2f}'.format(end_time - start_time), end='')
        return retval

    return wrapper


class ModelParams:

    def __init__(self, n_epochs=30, n_layers=2, vNetWidth=20, MC_samples=500000, batch_size0=65000, test_size=50000,
                 n_time_steps=96, S0=10, V0=0.04, rate=0.00, T=2):
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
        self.file_name = 'heston_r=0.0_S0=10_V0=0.04_kappa=1.5_theta=0.05_sigma=0.8_rho=-0.9'
        self.rate = float(self.file_name.split('_')[1].split('=')[1])
        self.S0 = float(self.file_name.split('_')[2].split('=')[1])
        self.V0 = float(self.file_name.split('_')[3].split('=')[1])
        self.kappa = float(self.file_name.split('_')[4].split('=')[1])
        self.theta = float(self.file_name.split('_')[5].split('=')[1])
        self.strikes_put = self.S0 * np.array([.55, .60, .65, .70, .75, .80, .85, .90, .95, 1.00])
        self.strikes_call = self.S0 * np.array([1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45])
        self.strikes = np.concatenate([self.strikes_put, self.strikes_call])
        self.VIX_strikes = np.sqrt(heston_VIX2(self.V0, self.kappa, self.theta)) * \
                           np.array([.5, .6, .7, .8, .9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
        self.monthly_step = n_time_steps // 24  # we have 24 maturities worth of Heston option prices
        self.maturities = [int(maturity) for maturity in
                           range(self.monthly_step, n_time_steps + self.monthly_step, self.monthly_step)]
        self.maturities = self.maturities[1:]


class NeuralNetCell(nn.Module):

    def __init__(self, dim, nOut, n_layers, vNetWidth, activation='relu', act_output='none', init_gain=1.75):
        super(NeuralNetCell, self).__init__()
        self.dim = dim
        self.nOut = nOut
        self.gain = init_gain

        if activation not in {'relu', 'tanh'}:
            raise ValueError("Unknown activation function {}".format(activation))
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()

        # if act_output not in {'relu', 'tanh', 'elu', 'softp', 'none'}:
        #     raise ValueError("Unknown activation function {}".format(act_output))
        if act_output == 'relu':
            self.out_activation = nn.ReLU()
        elif act_output == 'elu':
            self.out_activation = nn.ELU()
        elif act_output[:5] == 'softp':
            if act_output[5:] == '':
                self.out_activation = nn.Softplus()
            else:
                self.out_activation = nn.Softplus(beta=float(act_output[5:]))
        elif act_output == 'tanh':
            self.out_activation = nn.Tanh()
        elif act_output == 'sigmoid':
            self.out_activation = nn.Sigmoid()
        else:
            self.out_activation = nn.Identity()

        self.i_h = self.inputLayer(dim, vNetWidth)
        self.h_h = nn.ModuleList([self.hiddenLayerT1(vNetWidth, vNetWidth) for l in range(n_layers - 1)])
        self.h_o = self.outputLayer(vNetWidth, nOut, act='none')

    def initialization_decorator(f):
        def wrapped_f(*args, **kwargs):
            self = args[0]
            layer = f(*args, **kwargs)
            layer.apply(lambda m: self.init_weights(m, gain=self.gain))
            return layer

        return wrapped_f

    @staticmethod
    def init_weights(m, gain=1.75):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight, gain=gain)
            # nn.init.normal_(m.weight, mean=0.0, std=0.6)
            try:
                m.bias.data.fill_(0.01)
            except:
                pass

    @staticmethod
    def normalize_input(tensor):
        m = tensor.mean(dim=0).detach()
        std = tensor.std(dim=0).detach()
        m[0], std[std == 0.] = 0., 1.  # not de-meaning time, not normalizing by std if constant
        return (tensor - m) / std

    @initialization_decorator
    def inputLayer(self, nIn, nOut):
        layer = nn.Sequential(nn.Linear(nIn, nOut, bias=True),
                              # nn.BatchNorm1d(nOut, momentum=0.1),
                              self.activation)
        # layer.apply(self.init_weights)
        return layer

    @initialization_decorator
    def hiddenLayerT1(self, nIn, nOut):
        layer = nn.Sequential(nn.Linear(nIn, nOut, bias=True),
                              # nn.BatchNorm1d(nOut, momentum=0.01),
                              self.activation)
        # layer.apply(self.init_weights)
        return layer

    @initialization_decorator
    def outputLayer(self, nIn, nOut, act='none'):
        if act == 'none':
            layer = nn.Sequential(nn.Linear(nIn, nOut, bias=False))
        else:
            layer = nn.Sequential(nn.Linear(nIn, nOut, bias=False),
                                  # nn.BatchNorm1d(nOut, momentum=0.01),
                                  self.out_activation)
        # layer.apply(self.init_weights)
        return layer

    def forward(self, S):
        S = self.normalize_input(S)  # normalize the input
        h = self.i_h(S)
        for l in range(len(list(self.h_h))):
            h = list(self.h_h)[l](h)
        output = self.out_activation(self.h_o(h) + S.sum(dim=1, keepdim=True))  # skip connection
        return output


# Set up neural SDE class
class Net_SDE(nn.Module):

    def __init__(self, params):
        super(Net_SDE, self).__init__()
        self.params = params

        # Input to the coefficient (NN) will be (t,S_t,V_t)
        self.sigma = NeuralNetCell(dim=3, nOut=1, n_layers=params.n_layers, vNetWidth=params.vNetWidth,
                                   act_output='softp0.5',
                                   init_gain=1.)  # TODO: Maybe initialize all the parameters to be positive?
        # Input to the coefficient (NN) will be (tau,V_t)
        self.a = NeuralNetCell(dim=2, nOut=1, n_layers=params.n_layers, vNetWidth=params.vNetWidth,
                               act_output='none', init_gain=.25)
        # Input to the coefficient (NN) will be (tau,V_t)
        self.b = NeuralNetCell(dim=2, nOut=1, n_layers=params.n_layers, vNetWidth=params.vNetWidth,
                               act_output='softp', init_gain=1.)
        self.rho = nn.Parameter(-0.9 * torch.ones(1, device=device), requires_grad=True)

    def forward(self, W, scheme='reflection'):
        if scheme not in {'absorption', 'reflection'}:
            raise ValueError('Wrong discretization scheme.')
        dW, dB = W
        zeros, ones = torch.zeros(dW.shape[0], 1, device=device), torch.ones(dW.shape[0], 1, device=device)

        S_t, V_t = ones * self.params.S0, ones * self.params.V0

        price_call_mat = torch.zeros(len(self.params.maturities), len(self.params.strikes), device=device)
        price_put_mat = torch.zeros(len(self.params.maturities), len(self.params.strikes), device=device)
        price_VIX_call_mat = torch.zeros(len(self.params.maturities), len(self.params.VIX_strikes), device=device)
        price_VIX_put_mat = torch.zeros(len(self.params.maturities), len(self.params.VIX_strikes), device=device)
        Fwd_var_vec = torch.zeros(len(self.params.maturities), device=device)
        VIX_fwd_vec = torch.zeros(len(self.params.maturities), device=device)

        discount_factor = lambda t: torch.exp(-self.params.rate * t)
        VIX_delta = (self.params.time_grid[0 + self.params.monthly_step] - self.params.time_grid[0])

        # Euler-Maruyama S_t, V_t
        for idx, t in enumerate(self.params.time_grid[:-1], 1):
            input_sigma = torch.cat([ones * t, S_t, V_t], 1)
            input_vol = torch.cat([ones * t, V_t], 1).requires_grad_()

            # absorption ensures positive stock price
            # Increment tamed Euler-Mayurama scheme - Jentzen
            # Z1 = S_t * (self.params.rate * self.params.h) + self.sigma(input_sigma) * dW[:, idx - 1, None]
            # Z1 = S_t * (self.params.rate * self.params.h + self.sigma(input_sigma) * dW[:, idx - 1, None])
            Z1 = S_t * self.sigma(input_sigma)
            # S_new = S_t + Z1/(torch.max(ones, self.params.h * torch.abs(Z1)))
            S_new = S_t * (1 + self.params.rate * self.params.h) + Z1/(1 + 2.5 * torch.sqrt(self.params.h) * torch.abs(Z1)) * dW[:, idx - 1, None]
            # S_new = S_t + Z1
            S_t = torch.max(S_new, zeros) if scheme == 'absorption' else torch.abs(S_new)

            Z2 = self.a(input_vol) * self.params.h + self.b(input_vol) * (
                    torch.sqrt(1 - torch.pow(self.rho, 2)) * dB[:, idx - 1, None] + self.rho * dW[:, idx - 1, None])
            V_new = V_t + 0.05 * Z2/(torch.max(ones, self.params.h * torch.abs(Z2)))
            # V_new = V_t + 0.025 * Z2
            V_t = torch.max(V_new, zeros) if scheme == 'absorption' else torch.abs(V_new)

            if idx in self.params.maturities:
                idx_t = self.params.maturities.index(idx)

                # Use only first order for eval (no grad enabled), use second order for train (no grad disabled)
                # if V_t.requires_grad:
                #     Stochastic Taylor expansion of the expectation E[V_s|F_t] in the Second order
                #     input_vol.requires_grad_(True)
                #     a_grad = torch.autograd.grad(self.a(input_vol), input_vol, grad_outputs=ones,
                #                                  only_inputs=True, create_graph=True)[0]
                #     a_grad2 = torch.autograd.grad(a_grad, input_vol, grad_outputs=torch.cat([ones, ones], 1),
                #                                   only_inputs=True, create_graph=True)[0]  # TODO: always 0 ??
                #     first_order = self.a(input_vol) * (
                #                 self.params.time_grid[idx: idx + self.params.monthly_step] - ones * t)
                #     second_order = (a_grad[:, 0, None] * V_t + self.a(input_vol) * a_grad[:, 1, None] +
                #                     .5 * torch.pow(self.b(input_vol), 2) * a_grad2[:, 1, None]) * .5 * torch.pow(
                #                 self.params.time_grid[idx: idx + self.params.monthly_step] - ones * t, 2)
                #     VIX_t = torch.sqrt(self.params.h / VIX_delta * torch.abs(torch.sum(V_t + first_order + second_order,
                #                   1, keepdim=True))) * 100

                # else:
                # Stochastic Taylor expansion of the expectation E[V_s|F_t] in the First order
                # Ensure that VIX^2 is positive with the reflection scheme
                # VIX_t = torch.sqrt(self.params.h / VIX_delta * torch.abs(
                #     torch.sum(V_t + self.a(input_vol) * (
                #             self.params.time_grid[idx: idx + self.params.monthly_step] - ones * t),
                #             1, keepdim=True))) * 100

                VIX_t = torch.sqrt(torch.abs(V_t + 0.5 * self.a(input_vol) * VIX_delta)) * 100

                for idx_k, strike in enumerate(self.params.strikes):
                    price_call_mat[idx_t, idx_k] = (torch.max(S_t - strike, torch.zeros_like(S_t)) *
                                                    discount_factor(t)).mean()
                    price_put_mat[idx_t, idx_k] = (torch.max(strike - S_t, torch.zeros_like(S_t)) *
                                                   discount_factor(t)).mean()

                for idx_k, VIX_strike in enumerate(self.params.VIX_strikes):
                    price_VIX_call_mat[idx_t, idx_k] = (torch.max(VIX_t - VIX_strike, torch.zeros_like(VIX_t)) *
                                                    discount_factor(t)).mean()
                    price_VIX_put_mat[idx_t, idx_k] = (torch.max(VIX_strike - VIX_t, torch.zeros_like(VIX_t)) *
                                                   discount_factor(t)).mean()

                Fwd_var_vec[idx_t] = (V_t * discount_factor(t)).mean()
                VIX_fwd_vec[idx_t] = (VIX_t * discount_factor(t)).mean()

        return torch.cat([price_call_mat, price_put_mat], 0), torch.cat([price_VIX_call_mat, price_VIX_put_mat], 0),\
               Fwd_var_vec, VIX_fwd_vec


class SDE_tools:

    def __init__(self, params):
        self.params = params
        # ModelParams.__init__(self)

    @seed_decorator
    def generate_BMs(self, MC_samples, antithetics=False, **kwargs):
        # create BM increments
        dW = torch.sqrt(self.params.h) * torch.randn((MC_samples, len(self.params.time_grid)))
        dB = torch.sqrt(self.params.h) * torch.randn((MC_samples, len(self.params.time_grid)))
        if antithetics:
            return torch.cat([dW, -dW], 0), torch.cat([dB, -dB], 0)
        else:
            return dW, dB

    def generate_paths(self, model, W, scheme='reflection', detach_graph=False, no_grads=False):
        if scheme not in {'absorption', 'reflection'}:
            # absorption ensures positive volatility
            raise ValueError('Wrong discretization scheme.')

        # create BM increments
        dW, dB = W[0], W[1]

        MC_samples = dW.shape[0]
        zeros, ones = torch.zeros(MC_samples, 1, device=device), torch.ones(MC_samples, 1, device=device)
        S, V = torch.zeros((MC_samples, len(self.params.time_grid))), torch.zeros(
            (MC_samples, len(self.params.time_grid)))
        S[:, 0] = self.params.S0
        V[:, 0] = self.params.V0

        with torch.no_grad() if no_grads else ExitStack() as gs:
            # Euler-Maruyama S_t, V_t
            for idx, t in enumerate(self.params.time_grid[1:]):
                input_sigma = torch.cat([ones * t, S[:, idx, None], V[:, idx, None]], 1)
                input_vol = torch.cat([ones * t, V[:, idx, None]], 1)

                # absorption ensures positive stock price
                # Z1 = S[:, idx, None] * (
                #             self.params.rate * self.params.h) + model.sigma(input_sigma) * dW[:, idx - 1, None]
                # Z1 = S[:, idx, None] * (
                #             self.params.rate * self.params.h + model.sigma(input_sigma) * dW[:, idx - 1, None])
                Z1 = S[:, idx, None] * model.sigma(input_sigma)
                # S_new = S[:, idx, None] + Z1 / torch.max(ones, self.params.h * torch.abs(Z1))
                S_new = S[:, idx, None] * (1 + self.params.rate * self.params.h) + Z1 / (
                        1 + 2.5 * torch.sqrt(self.params.h) * torch.abs(Z1)) * dW[:, idx - 1, None]
                # S_new = S[:, idx, None] + Z1
                S[:, idx + 1] = (torch.max(S_new, zeros) if scheme == 'absorption' else torch.abs(S_new)).squeeze()

                Z2 = model.a(input_vol) * self.params.h + model.b(input_vol) * (
                        torch.sqrt(1 - torch.pow(model.rho, 2)) * dB[:, idx - 1, None] + model.rho * dW[:, idx - 1,
                                                                                                     None])
                V_new = V[:, idx, None] + 0.05 * Z2 / torch.max(ones, self.params.h * torch.abs(Z2))
                # V_new = V[:, idx, None] + 0.025 * Z2
                V[:, idx + 1] = (torch.max(V_new, zeros) if scheme == 'absorption' else torch.abs(V_new)).squeeze()

        if detach_graph:
            return S.detach(), V.detach()
        else:
            return S, V

    def true_fwd_var(self, maturity):
        """Generates Forward variance under Heston with params V_0, kappa, theta"""
        return self.params.V0 * torch.exp(-self.params.kappa * maturity) + \
               self.params.theta * (1. - torch.exp(-self.params.kappa * maturity))

    def calc_prices(self, S, mat=None, no_grads=False):
        discount_factor = lambda t: torch.exp(-self.params.rate * t)
        condition = S.squeeze().ndimension() <= 1
        if condition:
            price_call_mat = torch.zeros(1, len(self.params.strikes), device=device)
            price_put_mat = torch.zeros(1, len(self.params.strikes), device=device)
            S_T = S.view(-1, 1)
        else:
            price_call_mat = torch.zeros(len(self.params.maturities), len(self.params.strikes), device=device)
            price_put_mat = torch.zeros(len(self.params.maturities), len(self.params.strikes), device=device)
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

    def calc_VIX_prices(self, V, model, no_grads=False, detach=True):
        discount_factor = lambda t: torch.exp(-self.params.rate * t)

        price_call_mat = torch.zeros(len(self.params.maturities), len(self.params.VIX_strikes), device=device)
        price_put_mat = torch.zeros(len(self.params.maturities), len(self.params.VIX_strikes), device=device)

        VIX_delta = (self.params.time_grid[0 + self.params.monthly_step] - self.params.time_grid[0])
        V_T = V[:, self.params.maturities]
        # VIX_T = torch.sqrt(torch.abs(V_T + 0.5 * model.a(input_vol) * VIX_delta)) * 100  #TODO: !!!
        mat = self.params.time_grid[self.params.maturities]

        with torch.no_grad() if no_grads else ExitStack() as gs:
            for idx, strike in enumerate(self.params.strikes):
                price_call_mat[:, idx] = (torch.max(V_T - strike, torch.zeros_like(V_T)) *
                                          discount_factor(mat)).mean(dim=0)
                price_put_mat[:, idx] = (torch.max(strike - V_T, torch.zeros_like(V_T)) *
                                         discount_factor(mat)).mean(dim=0)
        if detach:
            return torch.cat([price_call_mat, price_put_mat], 0).detach()
        else:
            return torch.cat([price_call_mat, price_put_mat], 0)


class Trainer:

    def __init__(self, params, learning_rate=0.0025, schedule_val=0.985, clip_value=50, milestones=None, gamma=0.1):
        # super(Trainer, self).__init__()
        if milestones is None:
            self.milestones = [100, 200]

        self.params = params
        self.tools = SDE_tools(params)
        self.learning_rate = learning_rate
        self.schedule_val = schedule_val
        self.clip_value = clip_value

    @staticmethod
    @log_time
    def torch_backprop(loss):
        loss.backward()

    @staticmethod
    @log_time
    def torch_forward(model_train, BMs):
        return model_train(BMs)

    def loss_func(self, C, C_mkt, normalize=False):
        if normalize:
            return torch.mean(torch.pow((C - C_mkt) / (C_mkt + 0.001), 2))  # normalize by abs size/dvsn with 0
        else:
            return torch.mean(torch.pow(C - C_mkt, 2))

    def loss_sup_func(self, C, C_mkt):
        return torch.max(torch.abs(C - C_mkt))

    def loss_SPX_fwd(self, C, C_mkt, Fwd, Fwd_mkt, lam=0.5):
        # weighted loss of Forward variances and SPX options
        # N, M = len(self.params.maturities), len(self.params.strikes)
        C_loss, Fwd_loss = self.loss_func(C, C_mkt, False), self.loss_func(Fwd, Fwd_mkt, False)
        return lam * C_loss + (1 - lam) * Fwd_loss

    def batch_func(self, loss, batch_size_old=False):
        # Batch size as a function of upper bound on MC error
        beta = 0.01
        max_increase_ratio = 0.25
        if loss:
            batch_size_new = int(2.576 ** 2 / (4 * beta ** 2 * np.power(loss, 2)))
            if batch_size_old:
                increase = np.clip(batch_size_new - batch_size_old,
                                   a_min=-batch_size_old * max_increase_ratio,
                                   a_max=batch_size_old * max_increase_ratio)
                return min(batch_size_old + int(increase), self.params.batch_size0)
            else:
                return min(batch_size_new, self.params.batch_size0)
        else:
            return self.params.batch_size0

    def train_models(self, true_prices, true_VIX_prices, true_fwd_var, true_VIX_fwd):

        model = Net_SDE(self.params)

        # Perform gradient clipping
        # for p in model.parameters():
        #   p.register_hook(lambda grad: print(grad.max()))
          # p.register_hook(lambda grad: torch.clamp(grad, -self.clip_value, self.clip_value))

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate,
                                     eps=1e-08, amsgrad=True, betas=(0.9, 0.999),
                                     weight_decay=0)

        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=0.1)
        scheduler_func = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: self.schedule_val ** epoch)

        # Generate BMs
        W_test = self.tools.generate_BMs(self.params.test_size, antithetics=False,
                                         seed=0)
        W_test = W_test[0].to(device=device), W_test[1].to(device=device)

        S_test, V_test = self.tools.generate_paths(model, W_test, detach_graph=True)

        plt.figure(0)
        plt.plot(S_test[:100, :].cpu().T)
        # plt.show()
        plt.savefig('s_test.png')

        plt.figure(1)
        plt.plot(V_test[:100, :].cpu().T)
        # plt.show()
        plt.savefig('v_test.png')

        plt.figure(2)
        plt.plot(V_test[np.argmax(np.max(V_test.cpu().numpy(), 1)), :].cpu())
        # plt.show()
        plt.savefig('v_test_max.png')

        for epoch in range(self.params.n_epochs):
            # evaluate and print test error at the start of each epoch
            with torch.no_grad():
                model.eval()  # turn off Batch Normalization
                test_prices, test_VIX_prices, test_fwd_var, test_VIX_fwd = model(W_test)
                opt_test_loss = self.loss_func(test_prices, true_prices)
                VIX_test_loss = self.loss_func(test_VIX_prices, true_VIX_prices)
                fwd_test_loss = self.loss_func(test_fwd_var, true_fwd_var)
                VIX_fwd_test_loss = self.loss_func(test_VIX_fwd, true_VIX_fwd)
                test_loss = opt_test_loss + VIX_test_loss + fwd_test_loss + VIX_fwd_test_loss

            W = self.tools.generate_BMs(self.params.MC_samples, antithetics=False, seed=1)
            W = W[0].to(device=device), W[1].to(device=device)

            batch = 0
            batch_size = max(5000, self.batch_func(False))

            print('------------------------------------------------------------------------------')
            print('Epoch: {} ~ Batch size: {}'.format(epoch, batch_size))
            print('\tTest weighted MSE loss = {0:.4f}'.format(test_loss.item()))
            print('\tSPX option loss = {0:.6f}, VIX option loss = {1:.6f}, Forward volatility loss = {2:.6f}, '
                  'VIX Forward loss = {3:.6f}\n'.format(opt_test_loss, VIX_test_loss, fwd_test_loss, VIX_fwd_test_loss))

            while batch < W[0].shape[0]:
                timestart = time.time()
                W_batch = W[0][batch: min(batch + batch_size, W[0].shape[0]), :], \
                          W[1][batch: min(batch + batch_size, W[0].shape[0]), :]

                model.train()
                optimizer.zero_grad()

                print('\t\tForward pass ', end='')
                train_prices, train_VIX_prices, train_fwd_var, train_VIX_fwd = self.torch_forward(model, W_batch)
                opt_loss = self.loss_func(train_prices, true_prices)
                VIX_loss = self.loss_func(train_VIX_prices, true_VIX_prices)
                fwd_loss = self.loss_func(train_fwd_var, true_fwd_var)
                VIX_fwd_loss = self.loss_func(train_VIX_fwd, true_VIX_fwd)
                loss = opt_loss + VIX_loss + fwd_loss + VIX_fwd_loss

                print(' || Backward pass ', end='')
                self.torch_backprop(loss)
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                print('\n\t\tMSE loss = {0:.4f}'.format(loss.item()), end='')
                print('\t\t Total Time: {0:.2f}s & Batch size: {1}'.format(time.time() - timestart, batch_size))
                print('\t\t SPX option loss = {0:.6f}, VIX option loss = {1:.6f}, Forward volatility loss = {2:.6f}, '
                      'VIX Forward loss = {3:.6f}'.format(opt_loss, VIX_loss, fwd_loss, VIX_fwd_loss))

                # utils.plt_grads(model.diffusionV.h_h[0][0].weight.grad)
                scheduler_func.step()

                batch += batch_size
                batch_size = max(5000, self.batch_func(False, batch_size_old=batch_size))

            # scheduler.step()
        return model


if __name__ == "__main__":
    torch.manual_seed(1)
    params = ModelParams()
    start_time = time.time()
    timegrid = params.time_grid
    # Load market prices and set training target
    # ITM_call = torch.load('ITM_call.pt').to(device=device)
    # ITM_put = torch.load('ITM_put.pt').to(device=device)
    # OTM_call = torch.load('OTM_call.pt').to(device=device)
    # OTM_put = torch.load('OTM_put.pt').to(device=device)
    # C_mkt = torch.cat([ITM_call, OTM_call], 1)[:len(params.maturities), :]
    # P_mkt = torch.cat([OTM_put, ITM_put], 1)[:len(params.maturities), :]
    # SPX_mkt_prices = torch.cat([C_mkt, P_mkt], 0)

    # Remove the first maturity
    np_SPX_mkt_prices = np.concatenate([np.load(f'{params.file_name}_call.npy')[1:],
                                np.load(f'{params.file_name}_put.npy')[1:]], 0)
    SPX_mkt_prices = torch.from_numpy(np_SPX_mkt_prices).to(device=device)

    np_VIX_mkt_prices = np.concatenate([np.load(f'{params.file_name}_VIXcall.npy')[1:],
                                np.load(f'{params.file_name}_VIXput.npy')[1:]], 0)
    VIX_mkt_prices = torch.from_numpy(np_VIX_mkt_prices).to(device=device)

    Fwd_var_mkt = SDE_tools(params).true_fwd_var(params.time_grid[[0] + params.maturities])[1:]
    VIX_fwd_mkt = np.load(f'{params.file_name}_VIXfwd.npy')[1:]
    VIX_fwd_mkt = torch.from_numpy(VIX_fwd_mkt).to(device=device)

    model = Trainer(params).train_models(SPX_mkt_prices, VIX_mkt_prices, Fwd_var_mkt, VIX_fwd_mkt)
    print('--- MODEL TRAINING TIME: %d min %d s ---' % divmod(time.time() - start_time, 60))

    path = f'./models/{os.path.basename(__file__).split(".")[0]}_model_{time.strftime("%Y-%m-%d-%H%M%S")}.pth'
    torch.save(model.state_dict(), path)

    S, V = SDE_tools(params).generate_paths(model, SDE_tools(params).generate_BMs(60000), detach_graph=True,
                                            no_grads=True)

    model_SPX_prices = SDE_tools(params).calc_prices(S)
    model_SPX_IVs = calc_implied_vol.IV_lib(params.S0, params.rate, model_SPX_prices.cpu().numpy(),
                                            params.strikes, timegrid[params.maturities].cpu().numpy())
    calc_implied_vol.surf_plot_IVs(model_SPX_IVs, timegrid[params.maturities].cpu().numpy(), params.strikes,
                                   type='c', show=False)
    calc_implied_vol.surf_plot_IVs(model_SPX_IVs, timegrid[params.maturities].cpu().numpy(), params.strikes,
                                   type='p', show=False)

    plt.figure(3)
    plt.plot(V[:100, :].cpu().T)
    # plt.show()
    plt.savefig('v.png')

    plt.figure(4)
    plt.plot(V[np.argmax(np.max(V.cpu().numpy(), 1)), :].cpu())
    plt.savefig('v_max.png')

    plt.figure(5)
    plt.plot(S[:100, :].cpu().T)
    # plt.show()
    plt.savefig('S.png')

    print('done.')
