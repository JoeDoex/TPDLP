import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal
from scipy import special as ss
from utils.op import transition
import opt_einsum as oe

contract = oe.contract
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Time Series Season-Trend Decomposition
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class HiPPO_LegT(nn.Module):
    def __init__(self, N, dt=1.0, discretization='bilinear'):
        """
        N: the order of the HiPPO projection
        dt: discretization step size - should be roughly inverse to the length of the sequence
        """
        super(HiPPO_LegT,self).__init__()
        self.N = N
        # A, B = transition('lmu', N)
        A, B = transition('legs', N)
        C = np.ones((1, N))
        D = np.zeros((1,))
        # dt, discretization options
        A, B, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method=discretization)

        B = B.squeeze(-1)

        self.register_buffer('A', torch.Tensor(A).to(device))
        self.register_buffer('B', torch.Tensor(B).to(device))
        vals = np.arange(0.0, 1.0, dt)
        self.register_buffer('eval_matrix',  torch.Tensor(
            ss.eval_legendre(np.arange(N)[:, None], 1 - 2 * vals).T).to(device))

    def forward(self, inputs):  # torch.Size([128, 1, 1]) -
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection
        """

        c = torch.zeros(inputs.shape[:-1] + tuple([self.N])).to(device)  # torch.Size([1, 256])
        cs = []
        # print(inputs.shape)
        for f in inputs.permute([-1, 0, 1]):
            # print(f.shape)
            f = f.unsqueeze(-1)
            # f: [1,1]
            new = f @ self.B.unsqueeze(0) # [B, D, H, 256]
            c = F.linear(c, self.A) + new
            # print(c.shape)
            # c = [1,256] * [256,256] + [1, 256]
            cs.append(c)
        return torch.stack(cs, dim=0)

    def reconstruct(self, c):
        a = (self.eval_matrix @ c.unsqueeze(-1)).squeeze(-1)
        return (self.eval_matrix @ c.unsqueeze(-1)).squeeze(-1)


class Decomp_Forecast(nn.Module):
    def __init__(self, configs):
        super(Decomp_Forecast, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        if self.configs.ours:
            # b, s, f means b, f
            self.affine_weight = nn.Parameter(torch.ones(1, 1, configs.enc_in))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, configs.enc_in))

        self.multiscale = [1]
        self.window_size = configs.proj_degree
        self.legts = HiPPO_LegT(N=self.window_size, dt=1. / self.pred_len)
        self.mlp = nn.Linear(configs.proj_degree, configs.proj_degree)

    def forward(self, x_enc):
        # decomp init

        if self.configs.ours:
            means = x_enc.mean(1, keepdim=True).detach()
            # mean
            x_enc = x_enc - means
            # var
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x_enc /= stdev
            # affine
            x_enc = x_enc * self.affine_weight + self.affine_bias
        B, L, E = x_enc.shape
        seq_len = self.seq_len
        x_decs = []
        jump_dist = 0

        x_in_len = self.pred_len
        x_in = x_enc[:, -x_in_len:]
        legt = self.legts
        x_in_c = legt(x_in.transpose(1, 2)).permute([1, 2, 3, 0])[:, :, :, jump_dist:]
        out1 = self.mlp(x_in_c.permute(0, 1, 3, 2))
        out1 = out1.permute(0, 1, 3, 2)
        if self.seq_len >= self.pred_len:
            x_dec_c = out1.transpose(2, 3)[:, :, self.pred_len - 1 - jump_dist, :]
        else:
            x_dec_c = out1.transpose(2, 3)[:, :, -1, :]
        x_dec = x_dec_c @ (legt.eval_matrix[-self.pred_len:, :].T)
        x_dec = x_dec.permute(0, 2, 1)
        if self.configs.ours:
            x_dec = x_dec - self.affine_bias
            x_dec = x_dec / (self.affine_weight + 1e-10)
            x_dec = x_dec * stdev
            x_dec = x_dec + means

        return x_dec  # [B, L, D]


class Model(nn.Module):
    """
    TPDLP Model
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.channels = configs.enc_in

        self.Seasonal_F = Decomp_Forecast(configs)
        self.Trend_F = Decomp_Forecast(configs)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_output = self.Seasonal_F(seasonal_init)
        trend_output = self.Trend_F(trend_init)

        x = seasonal_output + trend_output
        return x  # to [Batch, Output length, Channel]
