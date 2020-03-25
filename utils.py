import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats
import time


class GradPlot:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.mat = None
        self.cbar = None

    def setup_plot(self, data):
        self.mat = self.ax.matshow(data)
        self.cbar = self.fig.colorbar(self.mat)
        self.cbar.ax.set_autoscale_on(True)
        self.fig.canvas.draw()
        plt.pause(0.0001)

    def replot_training(self, data):
        data = data.detach()
        if self.mat is None:
            self.setup_plot(data)
        self.fig.canvas.flush_events()
        self.mat.set_data(data)
        self.cbar.remove()
        # self.cbar = self.fig.colorbar(self.mat)
        # cbar_ticks = np.linspace(torch.min(data), torch.max(data), num=11, endpoint=True)
        # self.cbar.set_ticks(cbar_ticks)
        self.cbar = self.fig.colorbar(self.mat, ax=self.ax)
        self.fig.canvas.draw()
        plt.pause(0.0001)

    @staticmethod
    def plt_grads(grad, *args):
        plt.matshow(grad.detach().numpy())
        if len(args) > 0:
            plt.title(args[0])
        plt.colorbar()
        plt.show()
        time.sleep(0.1)


    # def setup_plot(data):
    #     plt.ion()
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     mat = ax.matshow(data)
    #     cbar = fig.colorbar(mat, ax=ax)
    #     cbar.ax.set_autoscale_on(True)
    #     fig.canvas.draw()
    #     plt.pause(0.0001)
    #     return fig, cbar, mat


    # def replot_training(fig, cbar, mat, data):
    #     fig.canvas.flush_events()
    #     cbar.remove()
    #     mat.set_data(data)
    #     cbar = fig.colorbar(mat)
    #     cbar_ticks = np.linspace(torch.min(data), torch.max(data), num=11, endpoint=True)
    #     cbar.set_ticks(cbar_ticks)
    #     fig.canvas.draw()
    #     plt.pause(0.0001)


def plot_density(Z):
    m = Z.mean()
    var = Z.var()
    pdf_x = np.linspace(np.min(Z), np.max(Z), 1000)
    pdf_y = 1.0 / np.sqrt(2 * np.pi * var) * np.exp(-0.5 * (pdf_x - m) ** 2 / var)
    plt.hist(Z, 100, density=True)
    plt.plot(pdf_x, pdf_y, 'k--')


def losses_reg(x, y):
    slopes = []
    intercepts = []
    stds = []
    for i in range(0, len(y), 50):
        slope, intercept, r_value, p_value, std = stats.linregress(np.array(range(len(y[i:min(i + 50, len(y))]))),
                                                                   y[i:min(i + 50, len(y))])
        slopes.append(slope)
        intercepts.append(intercept)
        stds.append(std)
    plt.plot(x, y)
    for idx, (slope, intercept, std) in enumerate(zip(slopes, intercepts, stds)):
        plt.plot(x[idx * 50: min(idx * 50 + 50, len(x))], x[idx * 50: min(idx * 50 + 50, len(x))] * slope + intercept)
        print(idx * 50, min(idx * 50, len(x)))
