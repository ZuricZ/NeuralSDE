import torch
import torch.nn
from torch.autograd import grad
import numpy as np
import operator


class backprop:
    def __init__(self, model, S, V, BMs):
        """ Detach from graph and turn reguires_grads=True, before running"""
        from Neural_SDE_manual_grad import ModelParams  # to avoid circular import
        self.model = model
        self.S = S
        self.V = V
        self.dW, self.dB = BMs
        self.param = ModelParams()
        self.rate = self.param.rate
        self.maturities = self.param.maturities
        self.strikes = self.param.strikes
        self.time_grid = self.param.time_grid
        self.h = self.param.h

        self.diffusion_coef = {key: value for key, value in self.model.named_parameters()
                               if key.startswith('diffusion.')}
        self.driftV_coef = [(key, value) for key, value in self.model.named_parameters()
                            if key.startswith('driftV.')]
        self.diffusionV_coef = [(key, value) for key, value in self.model.named_parameters()
                                if key.startswith('diffusionV.')]
        self.rho_coef = [(key, value) for key, value in self.model.named_parameters()
                         if key.startswith('rho.')]

    def NN_grad(self, net, inpt, param, grad_output, retain=True, clamp_grad=True):
        if clamp_grad:
            # clamp the gradients to they don't blow up
            return torch.clamp(grad(net(inpt), param, grad_outputs=grad_output, retain_graph=retain),
                               min=-1.125, max=1.125)
        else:  # Returns a tuple even if grad wrt to only one input !!
            return grad(net(inpt), param, grad_outputs=grad_output, retain_graph=retain)

    @staticmethod
    def mat_cum_prod(tens):
        out = torch.ones(tens.shape[0])
        for j in range(tens.shape[0]):
            out *= tens[j, :, :]
        return out

    @staticmethod
    def tuple_add(a, b):
        return tuple(map(operator.add, a, b))

    @staticmethod
    def tuple_mul(a, b):
        return tuple(map(operator.mul, a, b))

    @staticmethod
    def tuple_scalar_mul(a, b):
        return tuple(map(a.__mul__, b))

    @staticmethod
    def tuple_scalar_add(a, b):
        return tuple(map(a.__add__, b))

    def update_factor(self, old_factor, start, end, clamp_factor=True):
        dS_dTheta = 0  # TODO: either a tuple of 0 or the old factor !!
        dV_dPhi = 0
        dV_dNu = 0
        dV_dLambda = 0
        dS_dPhi = 0
        dS_dNu = 0
        dS_dLambda = 0
        for t in range(start, end):
            ones = torch.ones_like(self.S[:, t, None])
            input_S = torch.cat([ones * self.param.time_grid[t],
                                 self.S[:, t, None],
                                 self.V[:, t, None]], 1)
            input_V = torch.cat([ones * self.param.time_grid[t],
                                 self.V[:, t, None]], 1)

            dB_tilde = torch.sqrt(1-self.model.rho(t*ones))*self.dB[:, t, None] \
                                  + self.model.rho(t*ones)*self.dW[:, t, None]

            # TODO: tuple_scalar_mul is not appropriate !

            dSigma_dInputs = self.NN_grad(self.model.diffusion, input_S, input_S, self.dW[:, t, None])[0]  # t, S, V
            dSigma_dTheta = self.NN_grad(self.model.diffusion, input_S, self.model.diffusion.parameters(),
                                         self.dW[:, t, None])
            dS_dTheta = self.tuple_add(self.tuple_scalar_mul((1 + self.rate * self.h +
                                                                     dSigma_dInputs[:, 1, None]), dS_dTheta),
                                              dSigma_dTheta)

            dS_dPhi = self.tuple_add(self.tuple_scalar_mul(1 + self.rate * self.h
                                                                  + dSigma_dInputs[:, 1, None], dS_dPhi),
                                            self.tuple_scalar_mul(dSigma_dInputs[:, 2, None] * self.dW[:, t, None],
                                                                  dV_dPhi))

            dS_dNu = self.tuple_add(self.tuple_scalar_mul(1 + self.rate * self.h
                                                                  + dSigma_dInputs[:, 1, None], dS_dNu),
                                            self.tuple_scalar_mul(dSigma_dInputs[:, 2, None] * self.dW[:, t, None],
                                                                  dV_dNu))

            dS_dLambda = self.tuple_add(self.tuple_scalar_mul(1 + self.rate * self.h
                                                                 + dSigma_dInputs[:, 1, None], dS_dLambda),
                                           self.tuple_scalar_mul(dSigma_dInputs[:, 2, None] * self.dW[:, t, None],
                                                                 dV_dLambda))

            da_dInputs = self.NN_grad(self.model.driftV, input_V, input_V, ones*self.h)[0]  # t, V
            dv_dInputs = self.NN_grad(self.model.diffusionV, input_V, input_V, dB_tilde)[0]  # t, V
            da_dPhi = self.NN_grad(self.model.driftV, input_V, self.model.driftV.parameters(), self.h*ones)
            dV_dPhi = self.tuple_add(self.tuple_scalar_mul(1 + da_dInputs[:, 1, None]
                                                    + dv_dInputs[:, 1, None], dV_dPhi), da_dPhi)

            dv_dNu = self.NN_grad(self.model.diffusionV, input_V, self.model.diffusionV.parameters(), dB_tilde)
            dV_dNu = self.tuple_add(self.tuple_scalar_mul(1 + da_dInputs[:, 1, None]
                                                                 + dv_dInputs[:, 1, None], dV_dNu), dv_dNu)

            rho_eval = self.model.rho(t*ones)
            v_eval = self.model.diffusionV(input_V)
            Z = torch.sqrt(self.h) * (self.dW[:, t, None]
                           - 2 * torch.div(rho_eval, (torch.sqrt(1 - torch.pow(rho_eval, 2)))) * self.dB[:, t, None])
            dRho_dLambda = self.NN_grad(self.model.rho, input_V, self.model.rho.parameters(), v_eval*Z)

            dV_dLambda = self.tuple_add(self.tuple_scalar_add(1 + da_dInputs[:, 1, None]
                                               + dv_dInputs[:, 1, None], dV_dLambda), dRho_dLambda)

        ret_dict = {'dS_dTheta': dS_dTheta, 'dS_dPhi': dS_dPhi, 'dS_dNu': dS_dNu, 'dS_dLambda': dS_dLambda,
                    'dV_dPhi': dV_dPhi, 'dV_dNu': dV_dNu, 'dV_dLambda': dV_dLambda}
        if clamp_factor:
            return {key: factor.clamp(min=-1, max=1).detach() for key, factor in ret_dict}
        else:
            return {key: factor.detach() for key, factor in ret_dict}  # detaching for speed up

    def loss_grad(self, C_mkt, P_mkt):

        loss_gradient = {key: torch.zeros_like(param) for key, param in self.model.named_parameters()}

        discount_factor = lambda t: torch.exp(-self.rate * t)
        indicator = lambda C: (C >= 0).type(torch.int)

        factor = {'dS_dTheta': dS_dTheta, 'dS_dPhi': dS_dPhi, 'dS_dNu': dS_dNu, 'dS_dLambda': dS_dLambda,
                    'dV_dPhi': dV_dPhi, 'dV_dNu': dV_dNu, 'dV_dLambda': dV_dLambda}

        for idx_t, maturity in enumerate(self.maturities):

            factor = self.update_factor(factor, self.maturities[max(idx_t - 1, 0)], maturity)

            for idx_k, strike in enumerate(self.strikes):
                call_price_vec = (torch.max(self.S[:, maturity] - strike, torch.zeros_like(self.S[:, maturity]))
                                  * discount_factor(maturity)).mean(dim=0)
                put_price_vec = (torch.max(strike - self.S[:, maturity], torch.zeros_like(self.S[:, maturity]))
                                 * discount_factor(maturity)).mean(dim=0)

                diff_call_vec = (call_price_vec - C_mkt[idx_t, idx_k]).unsqueeze(1)
                diff_put_vec = (put_price_vec - P_mkt[idx_t, idx_k]).unsqueeze(1)

                # TODO: ^ this part can have more paths than the grad part (MC vs DL accuracy)

                call_term_output = (torch.max(self.S[:, maturity] - strike, torch.zeros_like(self.S[:, maturity]))
                                    * discount_factor(self.time_grid[maturity])) \
                                   * indicator(self.S[:, maturity] - strike) * self.dW[:, 0]

                put_term_output = (torch.max(strike - self.S[:, maturity], torch.zeros_like(self.S[:, maturity]))
                                   * discount_factor(self.time_grid[maturity])) \
                                  * indicator(strike - self.S[:, maturity]) * self.dW[:, 0]

                call_term_output, put_term_output = call_term_output.unsqueeze(1), put_term_output.unsqueeze(1)

                loss_gradient[key] += diff_call_vec[idx_t] * call_term_output * factor_call
                loss_gradient[key] += diff_put_vec[idx_t] * put_term_output * factor_put

        return {key: value / (len(self.strikes) + len(self.maturities) + self.S.shape[0]) for
                key, value in loss_gradient.items()}


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
