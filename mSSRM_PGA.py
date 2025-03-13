from typing import Tuple
import tqdm

import numpy as np
import pandas as pd
import torch


def mSSRM_PGA(m:int, iternum:int, tol:float, matR:torch.Tensor, vecmu:torch.Tensor) -> torch.Tensor:
    T, N = matR.shape
    RE = 100
    eI = torch.finfo(torch.float32).eps * torch.eye(N, device = matR.device)
    p = vecmu
    Q = (1/np.sqrt(T - 1)) * (matR - (1/T) * torch.ones((T, T), device = matR.device) @ matR)
    QeI = Q.T @ Q + eI
    alpha = 0.999 / QeI.norm(2)
    w = vecmu
    k = 1
    while k < iternum and RE > tol:
        w1 = w
        w_pre = w - alpha * (QeI @ w - p)
        w_pre = w_pre.clamp(min = 0)
        itw = torch.argsort(w_pre, descending = True)
        w = torch.zeros((N), device = matR.device)
        w[itw[:m]] = w_pre[itw[:m]]
        RE = (w - w1).norm(2) / w1.norm(2)
        k += 1

    if sum(w) == 0:
        return torch.zeros((N), device = matR.device)
    else:
        return w / sum(w)

def run_mSSRM_PGA(winsize:int, data:pd.DataFrame, m:int, device:torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    iternum = 1e4
    tol = 1e-5

    fullR = torch.tensor(data.values, dtype = torch.float, device = device)
    fullT, N = fullR.shape
    T_end = fullT
    all_w = torch.ones((fullT, N), device = device) / N
    CW = torch.zeros((T_end, 1), device = device)
    S = 1

    for t in tqdm.tqdm(range(T_end)):
        if t >= 5:
            if t < winsize:
                win_start = 0
            else:
                win_start = t - winsize
            win_end = t
            T = win_end - win_start + 1
            matR = fullR[win_start:win_end]
            vecmu = matR.mean(0).T
            w = mSSRM_PGA(m, iternum, tol, matR, vecmu)
            all_w[t, :] = w
            if w.sum():
                S = S * (fullR[t, :] + 1) @ w
        CW[t] = S

    A = CW[1:] / CW[:-1] - 1
    a = A.mean()
    b = A.std()
    sharpe = a/b

    return CW, all_w, sharpe