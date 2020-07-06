import torch
import torch.nn
from torch.autograd import grad
import numpy as np
import operator


class backprop:
    def __init__(self, params):
        """ Detach from graph and turn reguires_grads=True, before running"""
        self.params = params

    @staticmethod
    def sample_uniform(N):
        return torch.utils.data.SubsetRandomSampler(torch.range(0,N))

    def price_diff(self):
        pass


class Optimizers:
    @staticmethod
    def SGD(model, param_grads, gamma=2):
        sigma = model.diffusion
        with torch.no_grad():
            sigma.i_h[0].weight -= gamma * param_grads['i_h.0.weight']
            if sigma.i_h[0].bias is not None:
                sigma.i_h[0].bias -= gamma * param_grads['i_h.0.bias']
            for l in range(len(list(sigma.h_h))):
                sigma.h_h[l][0].weight -= gamma * param_grads[f'h_h.0.{l}.weight']
                sigma.h_h[l][0].bias -= gamma * param_grads[f'h_h.0.{l}.bias']
            sigma.h_o[0].weight -= gamma * param_grads['h_o.0.weight']
            if sigma.h_o[0].bias is not None:
                sigma.h_o[0].bias -= gamma * param_grads['h_o.0.bias']
        # for key, param_grad in param_grads:
        #     params[key] -= gamma * param_grads[key]
        # return model
