import numpy as np
import torch
import scipy.stats
import py_vollib
# from py_vollib.black_scholes import black_scholes as bs
from py_vollib.black.implied_volatility import implied_volatility as iv
# import pandas
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.cm as cm
from heston_VIX import heston_VIX2
import pickle


class IV_numpy:

    def __init__(self, S, K, T, r, q=0.0):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.n = scipy.stats.norm.pdf
        self.N = scipy.stats.norm.cdf
        self.MAX_ITERATIONS = 100
        self.PRECISION = 1.0e-5

    def bs_price(self, cp_flag, v):
        d1 = (np.log(self.S / self.K) + (self.r + v * v / 2.) * self.T) / (v * np.sqrt(self.T))
        d2 = d1 - v * np.sqrt(self.T)
        if cp_flag == 'c':
            price = self.S * np.exp(-self.q * self.T) * self.N(d1) - self.K * np.exp(-self.r * self.T) * self.N(d2)
        else:
            price = self.K * np.exp(-self.r * self.T) * self.N(-d2) - self.S * np.exp(-self.q * self.T) * self.N(-d1)
        return price

    def bs_vega(self, v):
        d1 = (np.log(self.S / self.K) + (self.r + v * v / 2.) * self.T) / (v * np.sqrt(self.T))
        return self.S * np.sqrt(self.T) * self.n(d1)

    def find_vol(self, target_value, call_put):
        sigma = 0.5*np.ones_like(target_value)
        for i in range(0, self.MAX_ITERATIONS):
            price = self.bs_price(call_put, sigma)
            vega = self.bs_vega(sigma)

            price = price
            diff = target_value - price  # our root

            if (abs(diff[~np.isnan(diff)]) < self.PRECISION).all():
                return sigma
            sigma[abs(diff) >= self.PRECISION] = sigma[abs(diff) >= self.PRECISION] + (diff / vega)[abs(diff) >= self.PRECISION]  # f(x) / f'(x)

        # value wasn't found, return best guess so far
        return sigma


class IV_torch:

    def __init__(self, S, K, T, r, q=0.0):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.n = lambda x: torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0])).log_prob(x).exp()
        self.N = lambda x: torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0])).cdf(x)
        self.MAX_ITERATIONS = 100
        self.PRECISION = 1.0e-5

    def bs_price(self, cp_flag, v):
        d1 = (torch.log(self.S / self.K) + (self.r + v * v / 2.) * self.T) / (v * np.sqrt(self.T))
        d2 = d1 - v * np.sqrt(self.T)
        if cp_flag == 'c':
            price = self.S * np.exp(-self.q * self.T) * self.N(d1) - self.K * np.exp(-self.r * self.T) * self.N(d2)
        else:
            price = self.K * np.exp(-self.r * self.T) * self.N(-d2) - self.S * np.exp(-self.q * self.T) * self.N(-d1)
        return price

    def bs_vega(self, v):
        d1 = (torch.log(self.S / self.K) + (self.r + v * v / 2.) * self.T) / (v * np.sqrt(self.T))
        return self.S * np.sqrt(self.T) * self.n(d1)

    def find_vol(self, target_value, call_put):
        sigma = 0.5*torch.ones_like(target_value)
        for i in range(0, self.MAX_ITERATIONS):
            price = self.bs_price(call_put, sigma)
            vega = self.bs_vega(sigma)

            diff = target_value - price  # our root

            if (abs(diff[~torch.isnan(diff)]) < self.PRECISION).all():
                return sigma
            sigma[abs(diff) >= self.PRECISION] = sigma[abs(diff) >= self.PRECISION] + (diff / vega)[
                abs(diff) >= self.PRECISION]  # f(x) / f'(x)

        # value wasn't found, return best guess so far
        return sigma


def IV_lib(S0, r, target, strikes, mat_times):
    # r = 0.02; S0 = 100
    IVs = np.zeros_like(target)
    mat = np.array(np.meshgrid(mat_times, strikes)).T.reshape(-1, 2)
    idx_half = target.shape[0]//2
    mat_c = np.c_[target[:idx_half, :].reshape(-1), mat]
    mat_p = np.c_[target[idx_half:, :].reshape(-1), mat]
    # mat = np.c_[mat, np.tile(mat_times, 2)]
    # mat = np.c_[mat, py_vollib.helpers.forward_price(S0, np.tile(mat_times, 2), r)]
    # for idx, mat in enumerate(mat_times):
    #     F = py_vollib.helpers.forward_price(S0, mat, r)
    #     IVs[idx, :] = np.apply_along_axis(lambda prices: iv(prices, F, strikes, r, mat, 'c'), 1,
    #                                       target[idx, :])
    #     IVs[idx+24, :] = np.apply_along_axis(lambda prices: iv(prices, F, strikes, r, mat, 'p'), 1,
    #                                       target[idx, :])

    def iv_with_exception_handling(price, F, K, r, T, flag):
        from py_lets_be_rational.exceptions import BelowIntrinsicException
        try:
            return iv(price, F, K, r, T, flag)
        except BelowIntrinsicException:
            return np.nan

    IVs[:idx_half, :] = np.apply_along_axis(lambda row: iv_with_exception_handling(row[0], py_vollib.helpers.forward_price(
        S0, row[1], r), row[2], r, row[1], 'c'), 1, mat_c).reshape(idx_half, -1)
    IVs[idx_half:, :] = np.apply_along_axis(lambda row: iv_with_exception_handling(row[0], py_vollib.helpers.forward_price(
        S0, row[1], r), row[2], r, row[1], 'p'), 1, mat_p).reshape(idx_half, -1)
    return IVs


def IV_lib_flag(S0, r, target, strikes, mat_times, flag='c'):
    mat = np.array(np.meshgrid(mat_times, strikes)).T.reshape(-1, 2)
    mat_c = np.c_[target.reshape(-1), mat]

    def iv_with_exception_handling(price, F, K, r, T, flag):
        from py_lets_be_rational.exceptions import BelowIntrinsicException
        try:
            return iv(price, F, K, r, T, flag)
        except BelowIntrinsicException:
            return np.nan

    IVs = np.apply_along_axis(lambda row: iv_with_exception_handling(row[0], py_vollib.helpers.forward_price(
        S0, row[1], r), row[2], r, row[1], flag), 1, mat_c).reshape(target.shape[0], -1)

    return IVs


def surf_plot_IVs(IVs, maturity_times, strikes, type='c', show=True):
    fig = plt.figure(np.random.randint(100) + 100)
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(maturity_times, strikes)

    if type == 'c':
        surf = ax.plot_surface(X, Y, IVs[:len(maturity_times), :].T, rstride=1, cstride=1, cmap=cm.coolwarm,
                               linewidth=0.1)
    else:
        surf = ax.plot_surface(X, Y, IVs[len(maturity_times):, :].T, rstride=1, cstride=1, cmap=cm.coolwarm,
                               linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    if show:
        plt.show()
    else:
        pickle.dump(fig, open(f'IV_{type}_surf.fig.pickle', 'wb'))
        # figx = pickle.load(open('FigureObject.fig.pickle', 'rb'))
        # figx.show()
        plt.savefig(f'IV_{type}_surf.png')


if __name__ == '__main__':
    # print(IV_numpy(586.08, 585.00, 0.10958, 0.0002).find_vol(np.array([17.0, 17.5, 18.0]), 'c'))
    # ITM_call = torch.load('ITM_call.pt')
    # ITM_put = torch.load('ITM_put.pt')
    # OTM_call = torch.load('OTM_call.pt')
    # OTM_put = torch.load('OTM_put.pt')
    # C_mkt = torch.cat([ITM_call, OTM_call], 1)
    # P_mkt = torch.cat([OTM_put, ITM_put], 1)
    # target_torch = torch.cat([C_mkt, P_mkt], 0)
    # target = target_torch.numpy()
    file_name = 'heston_r=0.0_S0=10_V0=0.04_kappa=1.5_theta=0.05_sigma=0.8_rho=-0.9'
    target = np.concatenate([np.load(f'{file_name}_VIXcall.npy'),
                             np.load(f'{file_name}_VIXput.npy')], 0)
    # target = np.delete(target, 9, axis=1)

    T = 2.0; n_steps = 48
    r = float(file_name.split('_')[1].split('=')[1])
    S0 = float(file_name.split('_')[2].split('=')[1])
    V0 = float(file_name.split('_')[3].split('=')[1])
    kappa = float(file_name.split('_')[4].split('=')[1])
    theta = float(file_name.split('_')[5].split('=')[1])

    # strikes = S0*np.array([.55, .60, .65, .70, .75, .80, .85, .90, .95, 1.00, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30,
    #                        1.35, 1.40, 1.45], dtype=np.float32)
    strikes = np.sqrt(heston_VIX2(V0, kappa, theta)) * np.array([.5, .6, .7, .8, .9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    maturities = list(range(2, n_steps + 1, 2))
    time_grid = np.linspace(0, T, n_steps + 1)

    IVs = np.zeros_like(target)

    for idx, mat in enumerate(maturities):
        IVs[idx, :] = IV_numpy(S0, strikes, time_grid[mat], r).find_vol(target[idx, :], 'c')
        IVs[idx + IVs.shape[0]//2, :] = IV_numpy(S0, strikes, time_grid[mat], r).find_vol(target[idx + 24, :], 'p')

    # IVs_torch = torch.zeros_like(target_torch)
    # for idx, mat in enumerate(maturities):
    #     IVs_torch[idx, :] = IV_torch(S0, torch.from_numpy(strikes), time_grid[mat], r).find_vol(
    #         target_torch[idx, :], 'c')
    #     IVs_torch[idx + IVs_torch.shape[0]//2, :] = IV_torch(S0, torch.from_numpy(strikes), time_grid[mat], r).find_vol(
    #         target_torch[idx + 24, :], 'p')

    IVs_ = IV_lib(S0, r, target, np.array(strikes), time_grid[maturities])

    H = IVs - IVs_
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('colorMap')
    plt.imshow(H)
    plt.colorbar(orientation='vertical')
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(time_grid[maturities], strikes)

    surf = ax.plot_surface(X, Y, IVs[:24, :].T, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
