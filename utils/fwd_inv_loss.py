import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("./utils/")
from utilities import *


myloss = LpLoss(size_average=False)

def INVoperator_net(s, x, t, Model):
    """Inverse operator network for resistivity reconstruction
    Args:
        s: MT response observations 
        x,t: positions
        Model: RDON architecture (contains trunk and branch networks)
    Returns:
        outputs: Reconstructed resistivity model
    """
    batch_size = s.shape[0]  

    # Combine spatial and frequency inputs
    y = torch.stack([x, t], dim=-1) 
    
    # Process through trunk network
    y = Model.trunk(y)  # Output
    D = y.shape[-1]  # Feature dimension size
    
    # Prepare for least squares solution
    Y_T = y.transpose(1, 2)  
    Y_T_Y = torch.matmul(Y_T, y)  
    Y_T_s = torch.matmul(Y_T, s.unsqueeze(-1)).squeeze(-1) 
    
    # Regularization for numerical stability
    epsilon = 1e-2
    eye = torch.eye(D, device=y.device).unsqueeze(0).expand(batch_size, -1, -1)  # Identity matrix
    Y_T_Y_reg = Y_T_Y + epsilon * eye  # Regularized matrix
    
    # Solve linear system: Y_T_Y_reg * b = Y_T_s
    b = torch.linalg.solve(Y_T_Y_reg, Y_T_s)  
    b = b.reshape(batch_size, 1, 32, 32)  # Reshape for branch network input
    
    # Inverse pass through reversible branch network
    outputs = Model.branch.backward(b)
    
    return outputs

def OP_residual_calculator(u, x, t, Guy_op, Model):
    """Forward operator residual calculator
    Args:
        u: Resistivity model 
        x,t: positions
        Guy_op: True MT responses 
        Model: RDON architecture
    Returns:
        loss: Forward prediction loss
    """
    Guy_op = Guy_op.squeeze(-1)  # Remove singleton dimension
    pred = Model(x, t, u)  # Forward prediction
    loss_op = myloss(pred, Guy_op)  # Compute prediction error
    return loss_op

def loss_inv(s, x, t, u, Model):
    """Inverse operator loss calculator
    Args:
        s: MT response observations
        x,t: positions
        u: True resistivity model
        Model: RDON architecture
    Returns:
        loss: Inversion reconstruction loss
    """
    s = s.squeeze(-1)  # Remove singleton dimension
    u_pred = INVoperator_net(s, x, t, Model)  # Reconstructed model
    loss_inv = myloss(u_pred, u)  # Reconstruction error
    return loss_inv