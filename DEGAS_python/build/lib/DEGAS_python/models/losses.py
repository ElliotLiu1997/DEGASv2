#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# For OT, we use code from: https://github.com/easezyc/deep-transfer-learning/blob/5e94d519b7bb7f94f0e43687aa4663aca18357de/MUDA/MFSAN/MFSAN_3src/mmd.py

import torch
import torch.nn as nn
import numpy as np

# refer to: https://github.com/easezyc/deep-transfer-learning/blob/5e94d519b7bb7f94f0e43687aa4663aca18357de/MUDA/MFSAN/MFSAN_3src/mmd.py
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val) / len(kernel_val)

# modify later to multi-kernel mmd
def MMD_loss(source, target, kernel_mul=2.0, kernel_num=20, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) -torch.mean(YX)
    return loss


# def c_index(pred, ytime, yevent):
#     '''Calculate concordance index to evaluate models.
#     Input:
#         pred: linear predictors from trained model.
#         ytime: true survival time from load_data().
#         yevent: true censoring status from load_data().
#     Output:
#         concordance_index: c-index (between 0 and 1).
#     '''
#     # sort the data
#     _, indices = torch.sort(ytime)
    
#     n_sample = len(ytime)
#     ytime_indicator = Rmatrix(ytime)
#     ytime_matrix = ytime_indicator - torch.diag(torch.diag(ytime_indicator))
#     ###T_i is uncensored
#     censor_idx = (yevent[indices] == 0).nonzero()
#     zeros = torch.zeros(n_sample)
#     ytime_matrix[censor_idx, :] = zeros
#     ###1 if pred_i < pred_j; 0.5 if pred_i = pred_j
#     pred_matrix = torch.zeros_like(ytime_matrix)
#     pred = pred[indices] # use the sorted index
#     for j in range(n_sample):
#         for i in range(n_sample):
#             if pred[i] < pred[j]:
#                 pred_matrix[j, i]  = 1
#             elif pred[i] == pred[j]: 
#                 pred_matrix[j, i] = 0.5


#     concord_matrix = pred_matrix.mul(ytime_matrix)
#     ###numerator
#     concord = torch.sum(concord_matrix)
#     ###denominator
#     epsilon = torch.sum(ytime_matrix) + 1e-7
#     ###c-index = numerator/denominator
#     concordance_index = torch.div(concord, epsilon)

#     return(concordance_index)


def Rmatrix(surv):
    surv = list(surv.cpu().detach().numpy())
    out = np.zeros([len(surv), len(surv)], dtype=int)
    for i in range(len(surv)):
        for j in range(len(surv)):
            out[i,j] = surv[j] >= surv[i]
    return torch.tensor(out)


# original degas loss function

def rank_loss(hazard, time, censor, device):
    loss = - torch.mean((hazard - torch.log(torch.sum(torch.exp(hazard).expand(time.shape[0], time.shape[0]) * Rmatrix(time).to(device), axis = 1))) * censor)
    return loss


######################################################################################################
########### Refer to: https://github.com/czifan/DeepSurv.pytorch/blob/master/networks.py #############
######################################################################################################

class NegativeLogLikelihood(nn.Module):
    def __init__(self, opt):
        super(NegativeLogLikelihood, self).__init__()
        self.device = torch.device('cuda:{}'.format(opt["gpu_id"])) if torch.cuda.is_available() else torch.device('cpu')  # get device name: CPU or GPU

    def forward(self, risk_pred, y, e, device): 
        mask = torch.ones(y.shape[0], y.shape[0]).to(self.device)
        mask[(y.T - y) > 0] = 0
        log_loss = torch.exp(risk_pred) * mask
        log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        neg_log_loss = -torch.sum((risk_pred-log_loss) * e) / torch.sum(e)
        return neg_log_loss

