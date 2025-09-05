import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("./scripts/")
from utilities import *

myloss = LpLoss(size_average=False)

def INVoperator_net(s, x, t, Model):
    # 获取 batch 大小
    batch_size = s.shape[0]  # 从 s 的第一个维度获取 batch 大小

    y = torch.stack([x, t], dim=-1)  # 形状为 (batch_size, num_points, 2)
    y = Model.trunk(y)   # 通过 trunk 网络，形状为 (batch_size, num_points, hidden_dim) ([50, 200, 100])
    D = y.shape[-1]  # 获取 D = hidden_dim = input_dim = output_dim = 100

    Y_T = y.transpose(1, 2)    # 转置y,形状为 (batch_size, hidden_dim, num_points) ([50, 100, 200])

    # 矩阵乘法计算 Y^T @ y
    Y_T_Y = torch.matmul(Y_T, y) # 形状为 (batch_size, hidden_dim, hidden_dim) (50, 100, 100)
    # 矩阵乘法，计算 Y^T @ s，
    Y_T_s = torch.matmul(Y_T, s.unsqueeze(-1)).squeeze(-1)   # 计算 Y^T @ s，形状为 (batch_size, hidden_dim)   # 矩阵乘法，形状为 (50, 100)

    # 添加正则化项
    epsilon = 1e-2
    eye = torch.eye(D, device=y.device).unsqueeze(0).expand(batch_size, -1, -1)  # 形状为 (batch_size, D, D) [50, 100, 100])
    
    Y_T_Y_reg = Y_T_Y + epsilon * eye  # 形状为 (batch_size, hidden_dim, hidden_dim)  ([50, 100, 100])

    # 求解线性方程组
    b = torch.linalg.solve(Y_T_Y_reg, Y_T_s)  # 形状为 (batch_size, hidden_dim)  ([50, 100])
    b = b.reshape(batch_size,1,32 ,32)  # 形状为 (batch_size, hidden_dim, 1) ([50, 100, 1])
    # 通过 branch_net 的逆运算计算输出
    outputs = Model.branch.backward(b)  # 形状为 (batch_size, hidden_dim)

    return outputs

# Define loss forward pass loss
def OP_residual_calculator(u, x, t,Guy_op, Model):
  Guy_op = Guy_op.squeeze(-1)
  pred = Model(x,t,u)
  loss_op = myloss(pred,Guy_op)
  loss = loss_op 
  return loss

# Define loss inverse pass loss
def loss_inv(s, x, t, u, Model):
  # Calculate the inverse operator
  s = s.squeeze(-1) # (50,200)
  u_pred = INVoperator_net(s, x, t, Model)
  loss_inv = myloss(u_pred, u)
  loss = loss_inv

  return loss