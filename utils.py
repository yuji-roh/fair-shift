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


def datasampling(xz_data, y_data, z_data, ex_weights, seed = 0):
    """Samples the data according to the example weights.

    Args:
        xz_data: A torch tensor indicating the input features.
        y_data: A torch tensor indicating the true label (1-D).
        z_data: A torch tensor indicating the sensitive attribute (1-D).
        ex_weights: A torch tensor indicating the example weights.
        seed: An integer indicating the random seed.

    Returns:
        selected data examples.
    """
    
    np.random.seed(seed)
    
    p_y1 = sum((y_data == 1.0).int()).float()/len(y_data)
    p_y0 = sum((y_data == -1.0).int()).float()/len(y_data)
    p_z1 = sum((z_data == 1.0).int()).float()/len(z_data)
    p_z0 = sum((z_data == 0.0).int()).float()/len(z_data)
    

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
    
    
    # Finds the original index
    z_index = {}
    y_index = {}
    yz_index = {}
        
    for tmp_z in z_item:
        z_index[tmp_z] = (z_mask[tmp_z] == 1).nonzero().squeeze()

    for tmp_y in y_item:
        y_index[tmp_y] = (y_mask[tmp_y] == 1).nonzero().squeeze()

    for tmp_yz in yz_tuple:
        yz_index[tmp_yz] = (yz_mask[tmp_yz] == 1).nonzero().squeeze()
    
    
    # Selects the final index
    selected_index = []
    max_weight = max(ex_weights)
    for tmp_yz in yz_tuple:
        selected_index.extend(np.random.choice(yz_index[tmp_yz].cpu(), int(len(yz_index[tmp_yz]) * ex_weights[yz_index[tmp_yz][0]]/max_weight), replace=False))
    
    return selected_index


def test_model(model_, X, y, s1):
    """Tests the performance of a model.

    Args:
        model_: A model to test.
        X: Input features of test data.
        y: True label (1-D) of test data.
        s1: Sensitive attribute (1-D) of test data.

    Returns:
        The test accuracy and the fairness metrics of the model.
    """
    
    model_.eval()
    
    y_hat = model_(X).squeeze()
    prediction = (y_hat > 0.0).int().squeeze()
    y = (y > 0.0).int()

    z_0_mask = (s1 == 0.0)
    z_1_mask = (s1 == 1.0)
    z_0 = int(torch.sum(z_0_mask))
    z_1 = int(torch.sum(z_1_mask))
    
    y_0_mask = (y == 0.0)
    y_1_mask = (y == 1.0)
    y_0 = int(torch.sum(y_0_mask))
    y_1 = int(torch.sum(y_1_mask))
    
    Pr_y_hat_1 = float(torch.sum((prediction == 1))) / (z_0 + z_1)
    
    Pr_y_hat_1_z_0 = float(torch.sum((prediction == 1)[z_0_mask])) / z_0
    Pr_y_hat_1_z_1 = float(torch.sum((prediction == 1)[z_1_mask])) / z_1
        
    
    y_1_z_0_mask = (y == 1.0) & (s1 == 0.0)
    y_1_z_1_mask = (y == 1.0) & (s1 == 1.0)
    y_1_z_0 = int(torch.sum(y_1_z_0_mask))
    y_1_z_1 = int(torch.sum(y_1_z_1_mask))
    
    Pr_y_hat_1_y_0 = float(torch.sum((prediction == 1)[y_0_mask])) / y_0
    Pr_y_hat_1_y_1 = float(torch.sum((prediction == 1)[y_1_mask])) / y_1
    
    Pr_y_hat_1_y_1_z_0 = float(torch.sum((prediction == 1)[y_1_z_0_mask])) / y_1_z_0
    Pr_y_hat_1_y_1_z_1 = float(torch.sum((prediction == 1)[y_1_z_1_mask])) / y_1_z_1
    
    y_0_z_0_mask = (y == 0.0) & (s1 == 0.0)
    y_0_z_1_mask = (y == 0.0) & (s1 == 1.0)
    y_0_z_0 = int(torch.sum(y_0_z_0_mask))
    y_0_z_1 = int(torch.sum(y_0_z_1_mask))

    Pr_y_hat_1_y_0_z_0 = float(torch.sum((prediction == 1)[y_0_z_0_mask])) / y_0_z_0
    Pr_y_hat_1_y_0_z_1 = float(torch.sum((prediction == 1)[y_0_z_1_mask])) / y_0_z_1
    
    recall = Pr_y_hat_1_y_1
    precision = float(torch.sum((prediction == 1)[y_1_mask])) / (int(torch.sum(prediction == 1)) +0.00001)
    
    y_hat_neq_y = float(torch.sum((prediction == y.int())))

    test_acc = torch.sum(prediction == y.int()).float() / len(y)
    test_f1 = 2 * recall * precision / (recall+precision+0.00001)
    
    min_dp = min(Pr_y_hat_1_z_0, Pr_y_hat_1_z_1) + 0.00001
    max_dp = max(Pr_y_hat_1_z_0, Pr_y_hat_1_z_1) + 0.00001
    min_eo_0 = min(Pr_y_hat_1_y_0_z_0, Pr_y_hat_1_y_0_z_1) + 0.00001
    max_eo_0 = max(Pr_y_hat_1_y_0_z_0, Pr_y_hat_1_y_0_z_1) + 0.00001
    min_eo_1 = min(Pr_y_hat_1_y_1_z_0, Pr_y_hat_1_y_1_z_1) + 0.00001
    max_eo_1 = max(Pr_y_hat_1_y_1_z_0, Pr_y_hat_1_y_1_z_1) + 0.00001
    
    DP = max(abs(Pr_y_hat_1_z_0 - Pr_y_hat_1), abs(Pr_y_hat_1_z_1 - Pr_y_hat_1))
    
    EO_Y_0 = max(abs(Pr_y_hat_1_y_0_z_0 - Pr_y_hat_1_y_0), abs(Pr_y_hat_1_y_0_z_1 - Pr_y_hat_1_y_0))
    EO_Y_1 = max(abs(Pr_y_hat_1_y_1_z_0 - Pr_y_hat_1_y_1), abs(Pr_y_hat_1_y_1_z_1 - Pr_y_hat_1_y_1))

    
    return {'Acc': test_acc.item(), 'DP_diff': DP, 'EO_Y0_diff': EO_Y_0, 'EO_Y1_diff': EO_Y_1, 'EqOdds_diff': max(EO_Y_0, EO_Y_1)}

