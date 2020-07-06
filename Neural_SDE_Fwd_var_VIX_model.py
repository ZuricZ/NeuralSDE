#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
# import torchviz
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

    def __init__(self, n_epochs=1, n_layers=2, vNetWidth=20, MC_samples=200000, batch_size0=30000, test_size=20000,
                 n_time_steps=96, S0=100, V0=0.04, rate=0.025, T=2):
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
        self.monthly_step = n_time_steps // 24  # we have 24 maturities worth of Heston option prices
        self.maturities = [int(maturity) for maturity in
                           range(self.monthly_step, n_time_steps + self.monthly_step, self.monthly_step)]
        self.S0 = S0
        self.V0 = V0
        self.rate = rate


class NeuralNetCell(nn.Module):

    def __init__(self, dim, nOut, n_layers, vNetWidth, activation="relu", act_output='none'):
        super(NeuralNetCell, self).__init__()
        self.dim = dim
        self.nOut = nOut

        if activation not in {'relu', 'tanh'}:
            raise ValueError("Unknown activation function {}".format(activation))
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()

        if act_output not in {'relu', 'tanh', 'elu', 'softp', 'none'}:
            raise ValueError("Unknown activation function {}".format(act_output))
        elif act_output == 'relu':
            self.out_activation = nn.ReLU()
        elif act_output == 'elu':
            self.out_activation = nn.ELU()
        elif act_output == 'softp':
            self.out_activation = nn.Softplus()
        elif act_output == 'tanh':
            self.out_activation = nn.Tanh()
        else:
            self.out_activation = nn.Identity()

        self.i_h = self.inputLayer(dim, vNetWidth)
        self.h_h = nn.ModuleList([self.hiddenLayerT1(vNetWidth, vNetWidth) for l in range(n_layers - 1)])
        self.h_o = self.outputLayer(vNetWidth, nOut, act='none')

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight, gain=0.05)
            # nn.init.normal_(m.weight, mean=0.0, std=0.05)
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

    def inputLayer(self, nIn, nOut):
        layer = nn.Sequential(nn.Linear(nIn, nOut, bias=True),
                              # nn.BatchNorm1d(nOut, momentum=0.1),
                              self.activation)
        layer.apply(self.init_weights)
        return layer

    def hiddenLayerT1(self, nIn, nOut):
        layer = nn.Sequential(nn.Linear(nIn, nOut, bias=True),
                              # nn.BatchNorm1d(nOut, momentum=0.01),
                              self.activation)
        layer.apply(self.init_weights)
        return layer

    def outputLayer(self, nIn, nOut, act='none'):
        if act == 'none':
            layer = nn.Sequential(nn.Linear(nIn, nOut, bias=False))
        else:
            layer = nn.Sequential(nn.Linear(nIn, nOut, bias=False), self.out_activation)
        layer.apply(self.init_weights)
        return layer

    def forward(self, S):
        S = self.normalize_input(S)  # normalize the input
        h = self.i_h(S)
        for l in range(len(list(self.h_h))):
            h = list(self.h_h)[l](h)
        output = self.out_activation(self.h_o(h) + S.sum(dim=1, keepdim=True))  # skip connection
        # nn.ConstantPad1d((0, 20 - 3), 0)(S).shape
        return output


# Set up neural SDE class
class Net_SDE(nn.Module):

    def __init__(self, params):
        super(Net_SDE, self).__init__()
        self.params = params

        # Input to the coefficient (NN) will be (t,S_t,xi_t)
        self.sigma = NeuralNetCell(dim=3, nOut=1, n_layers=params.n_layers, vNetWidth=params.vNetWidth,
                                   act_output='softp')  # We restrict diffusion coefficients to be positive  # TODO: Maybe initialize all the parameters to be positive?
        # Input to the coefficient (NN) will be (t,T,xi_t)
        self.nu = NeuralNetCell(dim=2, nOut=1, n_layers=params.n_layers, vNetWidth=params.vNetWidth,
                                act_output='softp')  # We restrict diffusion coefficients to be positive
        # self.f = NeuralNetCell(dim=1, nOut=1, n_layers=1, vNetWidth=2, act_output='softp')
        # self.rho = NeuralNetCell(dim=1, nOut=1, n_layers=1, vNetWidth=2,1
        #                          act_output='tanh')  # restrict rho to [-1,1]

        self.rho = nn.Parameter(-0.9*torch.ones(1, device=device), requires_grad=True)

    def forward(self, W, scheme='reflection'):
        if scheme not in {'absorption', 'reflection'}:
            raise ValueError('Wrong discretization scheme.')
        dW, dB = W
        zeros, ones = torch.zeros(dW.shape[0], 1, device=device), torch.ones(dW.shape[0], 1, device=device)

        S_t, xi_t = ones * self.params.S0, ones * self.params.V0
        B_lambda = zeros
        # dB_lambda = torch.zeros_like(dW, device=device)
        # BM_integral = zeros

        price_call_mat = torch.zeros(len(self.params.maturities), len(self.params.strikes), device=device)
        price_put_mat = torch.zeros(len(self.params.maturities), len(self.params.strikes), device=device)
        Fwd_var_vec = torch.zeros(len(self.params.maturities)+1, device=device)
        VIX_fwd_vec = torch.zeros(len(self.params.maturities)+1, device=device)

        discount_factor = lambda t: torch.exp(-self.params.rate * t)
        VIX_delta = (self.params.time_grid[0 + self.params.monthly_step] - self.params.time_grid[0])

        # Brownian motion increments
        # for idx, t in enumerate(self.params.time_grid[:-1], 1):
        #     dB_new = torch.sqrt(1 - torch.pow(self.rho, 2)) * dB[:, idx-1, None] \
        #             + self.rho * dW[:, idx-1, None]
        #     dB_lambda[:, idx] = dB_new.squeeze()

        # Euler-Maruyama S_t, V_t
        for idx, t in enumerate(self.params.time_grid[:-1], 1):
            input_sigma = torch.cat([ones * t, S_t, xi_t], 1)
            # absorption ensures positive stock price
            S_new = S_t * (1 + self.params.rate * self.params.h) + self.sigma(input_sigma) * dW[:, idx - 1, None]
            S_t = torch.max(S_new, zeros)

            # BM_new = BM_integral + self.f(t*ones)*(torch.sqrt(1 - torch.pow(self.rho, 2)) * dB[:, idx - 1, None]
            #         + self.rho * dW[:, idx - 1, None])
            # BM_integral = BM_new

            input_nu = torch.cat([zeros, B_lambda], 1)
            xi_t = self.nu(input_nu)

            if idx in [0] + self.params.maturities:
                idx_t = self.params.maturities.index(idx)
                inputs_VIX = torch.zeros(dW.shape[0], self.params.monthly_step, device=device)

                for idx_s, s in enumerate(self.params.time_grid[idx: idx + self.params.monthly_step]):
                    inputs_VIX[:, idx_s] = self.nu(torch.cat([s*ones, B_lambda], 1)).squeeze()

                VIX_t = torch.sqrt(self.params.h / VIX_delta * torch.sum(inputs_VIX, 1, keepdim=True))

                for idx_k, strike in enumerate(self.params.strikes):
                    price_call_mat[idx_t, idx_k] = (torch.max(S_t - strike, torch.zeros_like(S_t)) *
                                                    discount_factor(t)).mean()
                    price_put_mat[idx_t, idx_k] = (torch.max(strike - S_t, torch.zeros_like(S_t)) *
                                                   discount_factor(t)).mean()

                Fwd_var_vec[idx_t] = xi_t.mean()
                VIX_fwd_vec[idx_t] = VIX_t.mean()

            B_new = B_lambda + torch.sqrt(1 - torch.pow(self.rho, 2)) * dB[:, idx - 1, None] \
                    + self.rho * dW[:, idx - 1, None]
            B_lambda = B_new

        return torch.cat([price_call_mat, price_put_mat], 0), Fwd_var_vec, VIX_fwd_vec


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
        B_lambda = zeros

        with torch.no_grad() if no_grads else ExitStack() as gs:
            # Euler-Maruyama S_t, V_t
            for idx, t in enumerate(self.params.time_grid[1:]):
                input_sigma = torch.cat([ones * t, S[:, idx, None], V[:, idx, None]], 1)

                input_nu = torch.cat([zeros, B_lambda], 1)

                # absorption ensures positive stock price
                V[:, idx] = model.nu(input_nu).squeeze()
                S[:, idx + 1] = torch.max(S[:, idx] * (1 + self.params.rate * self.params.h) +
                                          model.sigma(input_sigma).squeeze() * dW[:, idx], zeros.squeeze())

                # V[:, idx + 1] = torch.max(V_temp.squeeze(), zeros.squeeze()) \
                #     if scheme == 'absorption' else torch.abs(V_temp.squeeze())

                B_lambda += torch.sqrt(1 - torch.pow(model.rho, 2)) * dB[:, idx, None] \
                            + model.rho * dW[:, idx, None]

                # B_lambda += model.f(t*ones)*torch.sqrt(1 - torch.pow(model.rho, 2)) * dB[:, idx, None] \
                #             + model.rho * dW[:, idx, None]

            input_nu = torch.cat([zeros, B_lambda], 1)
            V[:, idx+1] = model.nu(input_nu).squeeze()

        if detach_graph:
            return S.detach(), V.detach()
        else:
            return S, V

    def true_fwd_var(self, maturity):
        """Generates Forward variance under Heston with params V_0 = 0.04, kappa = 1.5, theta = 0.04"""
        kappa = 1.5
        theta = self.params.V0
        return self.params.V0 * torch.exp(-kappa * maturity) + theta * (1. - torch.exp(-kappa * maturity))

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


class Trainer:

    def __init__(self, params, learning_rate=0.01, clip_value=0.5, milestones=None, gamma=0.1):
        # super(Trainer, self).__init__()
        if milestones is None:
            milestones = [100, 200]

        self.params = params
        self.tools = SDE_tools(params)
        self.learning_rate = learning_rate
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

    def reg_func(self, nu_output, i_epoch):
        epsilon = lambda i: 0.01 * np.power(i / self.params.n_epochs + 0.1, -2)
        return (torch.max(-epsilon(i_epoch) - (nu_output - 1), torch.zeros_like(nu_output)) +
                torch.max((nu_output - 1) - epsilon(i_epoch), nu_output)).mean()

    def loss_SPX_fwd(self, C, C_mkt, Fwd, Fwd_mkt, lam=0.5):
        # weighted loss of Forward variances and SPX options
        # N, M = len(self.params.maturities), len(self.params.strikes)
        C_loss, Fwd_loss = self.loss_func(C, C_mkt, False), self.loss_func(Fwd, Fwd_mkt, False)
        return lam * C_loss + (1 - lam) * Fwd_loss

    def batch_func(self, loss, batch_size_old=False):
        # Batch size as a function of upper bound on MC error
        beta = 0.0075
        max_increase_ratio = 0.1
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

    # noinspection PyTypeChecker
    def train_models(self, true_prices, true_fwd_var, true_VIX_fwd):

        model = Net_SDE(self.params)

        # perform gradient clipping
        # for p in model.parameters():
            # p.register_hook(lambda grad: print(grad.max()))
            # p.register_hook(lambda grad: torch.clamp(grad, -self.clip_value, self.clip_value))

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, eps=1e-08, amsgrad=True, betas=(0.9, 0.999),
                                     weight_decay=0)
        # optimizer= torch.optim.Rprop(model.parameters(), lr=0.001, etas=(0.5, 1.2), step_sizes=(1e-07, 1))

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 7], gamma=0.1)
        # scheduler_func = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.985 ** epoch)

        # loss_lambda = lambda i_epoch: 10.*np.power(i_epoch/self.params.n_epochs + 1, -4) + 0.5  # 10*(x+1)^(-4) + 0.5
        loss_lambda = lambda i_epoch: 0.5

        # fix the seeds for reproducibility and generate antithetics and pass to torch
        W_test = self.tools.generate_BMs(self.params.test_size, antithetics=True,
                                         seed=0)  # not really using antithetics since they are not in the same batch
        W_test = W_test[0].to(device=device), W_test[1].to(device=device)
        W = self.tools.generate_BMs(self.params.MC_samples, antithetics=True, seed=1)
        W = W[0].to(device=device), W[1].to(device=device)

        for epoch in range(self.params.n_epochs):
            # evaluate and print test error at the start of each epoch
            with torch.no_grad():
                model.eval()  # turn off Batch Normalization
                test_prices, test_fwd_var, test_VIX_fwd = model(W_test)
                opt_test_loss = self.loss_func(test_prices, true_prices)
                fwd_test_loss = self.loss_func(test_fwd_var, true_fwd_var)
                VIX_fwd_test_loss = self.loss_func(test_VIX_fwd, true_VIX_fwd)
                test_loss = opt_test_loss + fwd_test_loss + VIX_fwd_test_loss

            # S_test, V_test = self.tools.generate_paths(model, W_test, no_grads=True, detach_graph=True)
            # test_prices = self.tools.calc_prices(S_test)
            # test_loss = self.loss_func(test_prices, target)

            batch = 0
            batch_size = max(500, self.batch_func(False))

            print('------------------------------------------------------------------------------')
            print('Epoch: {} ~ Batch size: {}'.format(epoch, batch_size))
            print('\tTest weighted MSE loss (no initial cond) = {0:.4f}'.format(test_loss.item()))
            print('\tSPX option loss = {0:.6f}, Forward volatility loss = {1:.6f}, VIX Forward loss = {2:.6f}'.format(
                opt_test_loss, fwd_test_loss, VIX_fwd_test_loss))

            while batch < W[0].shape[0]:
                timestart = time.time()
                W_batch = W[0][batch: min(batch + batch_size, W[0].shape[0]), :], \
                          W[1][batch: min(batch + batch_size, W[0].shape[0]), :]

                model.train()
                optimizer.zero_grad()

                print('\t\tForward pass ', end='')
                train_prices, train_fwd_var, train_VIX_fwd = self.torch_forward(model, W_batch)
                # nu_0s = model.nu(self.params.time_grid[[0] + self.params.maturities].unsqueeze(1))
                nu_input = torch.cat([2*torch.rand(1000, 1), torch.randn(1000, 1)], 1).requires_grad_()
                opt_loss = self.loss_func(train_prices, true_prices)
                fwd_loss = self.loss_func(train_fwd_var, true_fwd_var)
                VIX_fwd_loss = self.loss_func(train_VIX_fwd, true_VIX_fwd)
                # grad_loss = torch.sum(torch.pow(torch.autograd.grad(model.nu(nu_input), nu_input, grad_outputs=torch.ones(1000, 1),
                #                                 create_graph=True)[0] - 1, 2))
                grad_input = torch.autograd.grad(model.nu(nu_input), nu_input, grad_outputs=torch.ones(1000, 1),
                                                create_graph=True)[0]
                grad_input2 = torch.autograd.grad(grad_input, nu_input, grad_outputs=torch.ones(1000, 2),
                                                create_graph=True)[0]
                martingale_loss = torch.sum(torch.pow(grad_input[:, 0] - grad_input2[:, 1], 2))
                loss = opt_loss + 10*fwd_loss + 10*VIX_fwd_loss + 0.01*martingale_loss
                # + self.loss_sup_func(model.nu(nu_input), torch.ones(100, 1)*self.params.V0)

                print(' || Backward pass ', end='')
                self.torch_backprop(loss)
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                print('\n\t\tMSE loss = {0:.4f}'.format(loss.item()), end='')
                print('\t\t Total Time: {0:.2f}s & Batch size: {1}'.format(time.time() - timestart, batch_size))
                print('\t\t SPX option loss = {0:.6f}, Forward volatility loss = {1:.6f}, VIX Forward loss = {2:.6f}'.format(
                        opt_loss, fwd_loss, VIX_fwd_loss))

                # utils.plt_grads(model.diffusionV.h_h[0][0].weight.grad)
                # scheduler_func.step()
                batch += batch_size
                # batch_size = max(500, self.batch_func(opt_loss.item(), batch_size_old=batch_size))
                batch_size = self.batch_func(False)

        scheduler.step()
        return model


if __name__ == "__main__":
    torch.manual_seed(1)
    params = ModelParams()
    start_time = time.time()
    timegrid = params.time_grid
    # Load market prices and set training target
    ITM_call = torch.load('ITM_call.pt').to(device=device)
    ITM_put = torch.load('ITM_put.pt').to(device=device)
    OTM_call = torch.load('OTM_call.pt').to(device=device)
    OTM_put = torch.load('OTM_put.pt').to(device=device)
    C_mkt = torch.cat([ITM_call, OTM_call], 1)[:len(params.maturities), :]
    P_mkt = torch.cat([OTM_put, ITM_put], 1)[:len(params.maturities), :]
    target = torch.cat([C_mkt, P_mkt], 0)

    Fwd_var_mkt = SDE_tools(params).true_fwd_var(params.time_grid[[0]+params.maturities])
    VIX_fwd_mkt = np.load('VIX_heston_forwards.npy')/100
    VIX_fwd_mkt = np.insert(VIX_fwd_mkt, 0, params.V0)
    VIX_fwd_mkt = torch.from_numpy(VIX_fwd_mkt)

    model = Trainer(params).train_models(target, Fwd_var_mkt, VIX_fwd_mkt)
    print('--- MODEL TRAINING TIME: %d min %d s ---' % divmod(time.time() - start_time, 60))

    path = f'./models/{os.path.basename(__file__).split(".")[0]}_model_{time.strftime("%Y-%m-%d-%H%M%S")}.pth'
    torch.save(model.state_dict(), path)

    S, V = SDE_tools(params).generate_paths(model, SDE_tools(params).generate_BMs(5000), detach_graph=True,
                                            no_grads=True)

    print(torch.pow(V[:, :].mean(dim=0) - 0.04, 2).mean())
    print(torch.pow(SDE_tools(params).calc_prices(S) - target, 2).mean())
    plt.plot(V[:10, :].T)
    plt.show()

    print('done.')
