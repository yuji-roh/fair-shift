import sys, os
import numpy as np
import math
import random 
import itertools

import torch
import torch.nn as nn

def correlation_reweighting(xz_data, y_data, z_data, w, w_new):
    """Finds example weights according to the new weight.

    Args:
        xz_data: A torch tensor indicating the input features.
        y_data: A torch tensor indicating the true label (1-D).
        z_data: A torch tensor indicating the sensitive attribute (1-D).
        w: A list indicating the original data ratio for each (y, z)-class.
        w_new: A list indicating the new data ratio for each (y, z)-class.

    Returns:
        example weights.
    """
    
    p_y1 = sum((y_data == 1.0).int()).float()/len(y_data)
    p_y0 = sum((y_data == -1.0).int()).float()/len(y_data)
    p_z1 = sum((z_data == 1.0).int()).float()/len(z_data)
    p_z0 = sum((z_data == 0.0).int()).float()/len(z_data)
    
    #re-weight
    # Takes the unique values of the tensors
    z_item = list(set(z_data.tolist()))
    y_item = list(set(y_data.tolist()))

    yz_tuple = list(itertools.product(y_item, z_item))

    # Makes masks
    z_mask = {}
    y_mask = {}
    yz_mask = {}

    for tmp_z in z_item:
        z_mask[tmp_z] = (z_data == tmp_z)

    for tmp_y in y_item:
        y_mask[tmp_y] = (y_data == tmp_y)

    for tmp_yz in yz_tuple:
        yz_mask[tmp_yz] = (y_data == tmp_yz[0]) & (z_data == tmp_yz[1])

    len_y0_z0 = sum(yz_mask[-1,0].int())
    len_y1_z0 = sum(yz_mask[1,0].int())
    len_y0_z1 = sum(yz_mask[-1,1].int())
    len_y1_z1 = sum(yz_mask[1,1].int())


    ex_weight = []

    for i in range(len(y_data)):
        if yz_mask[-1,0][i] == 1:
            ex_weight.append(w_new[3]/w[3])
        elif yz_mask[1,0][i] == 1:
            ex_weight.append(w_new[1]/w[1])
        elif yz_mask[-1,1][i] == 1:
            ex_weight.append(w_new[2]/w[2])
        else:
            ex_weight.append(w_new[0]/w[0]) 

    ex_weight = torch.FloatTensor(ex_weight)
    
    return ex_weight



