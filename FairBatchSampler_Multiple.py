import sys, os
import numpy as np
import math
import random
import itertools
import copy

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import torch

class CustomDataset(Dataset):
    """Custom Dataset.

    Attributes:
        x: A PyTorch tensor for x features of data.
        y: A PyTorch tensor for y features (true labels) of data.
        z: A PyTorch tensor for z features (sensitive attributes) of data.
    """
    def __init__(self, x_tensor, y_tensor, z_tensor):
        """Initializes the dataset with torch tensors."""
        
        self.x = x_tensor
        self.y = y_tensor
        self.z = z_tensor
        
    def __getitem__(self, index):
        """Returns the selected data based on the index information."""
        
        return (self.x[index], self.y[index], self.z[index])

    def __len__(self):
        """Returns the length of data."""
        
        return len(self.x)


class FairBatch(Sampler):
    """FairBatch (Sampler in DataLoader).
    
    This class is for implementing the lambda adjustment and batch selection of FairBatch.

    Attributes:
        model: A model containing the intermediate states of the training.
        x_, y_, z_data: Tensor-based train data.
        alpha: A positive number for step size that used in the lambda adjustment.
        fairness_type: A string indicating the target fairness type 
                       among original, demographic parity (dp), equal opportunity (eqopp), and equalized odds (eqodds).
        knob: A real number indicating the importance between two fairness metrics.
        replacement: A boolean indicating whether a batch consists of data with or without replacement.
        N: An integer counting the size of data.
        batch_size: An integer for the size of a batch.
        batch_num: An integer for total number of batches in an epoch.
        y_, z_item: Lists that contains the unique values of the y_data and z_data, respectively.
        yz_tuple: Lists for pairs of y_item and z_item.
        y_, z_, yz_mask: Dictionaries utilizing as array masks.
        y_, z_, yz_index: Dictionaries containing the index of each class.
        y_, z_, yz_len: Dictionaries containing the length information.
        S: A dictionary containing the default size of each class in a batch.
        lb1, lb2: (0~1) real numbers indicating the lambda values in FairBatch.
        
    """
    def __init__(self, model, x_tensor, y_tensor, z_tensor, batch_size, alpha, target_fairness, knob = 1, grid = [], replacement = False, seed = 0):
        """Initializes FairBatch."""
        
        self.grid = grid
        
        self.model = model
        
        np.random.seed(seed)
        random.seed(seed)
        
        self.x_data = x_tensor
        self.y_data = y_tensor
        self.z_data = z_tensor
        
        self.useCuda = y_tensor.is_cuda
        
        self.alpha = alpha
        self.fairness_type = target_fairness
        self.knob = knob
        self.replacement = replacement
        
        self.N = len(z_tensor)
        
        self.batch_size = batch_size
        self.batch_num = int(len(self.y_data) / self.batch_size)
        
        # Takes the unique values of the tensors
        self.z_item = list(set(z_tensor.tolist()))
        self.y_item = list(set(y_tensor.tolist()))
        
        self.yz_tuple = list(itertools.product(self.y_item, self.z_item))
        
        # Makes masks
        self.z_mask = {}
        self.y_mask = {}
        self.yz_mask = {}
        
        for tmp_z in self.z_item:
            self.z_mask[tmp_z] = (self.z_data == tmp_z)
            
        for tmp_y in self.y_item:
            self.y_mask[tmp_y] = (self.y_data == tmp_y)
            
        for tmp_yz in self.yz_tuple:
            self.yz_mask[tmp_yz] = (self.y_data == tmp_yz[0]) & (self.z_data == tmp_yz[1])
        

        # Finds the index
        self.z_index = {}
        self.y_index = {}
        self.yz_index = {}
        
        for tmp_z in self.z_item:
            self.z_index[tmp_z] = (self.z_mask[tmp_z] == 1).nonzero().squeeze()
            
        for tmp_y in self.y_item:
            self.y_index[tmp_y] = (self.y_mask[tmp_y] == 1).nonzero().squeeze()
        
        for tmp_yz in self.yz_tuple:
            self.yz_index[tmp_yz] = (self.yz_mask[tmp_yz] == 1).nonzero().squeeze()
            
        # Length information
        self.z_len = {}
        self.y_len = {}
        self.yz_len = {}
        
        for tmp_z in self.z_item:
            self.z_len[tmp_z] = len(self.z_index[tmp_z])
            
        for tmp_y in self.y_item:
            self.y_len[tmp_y] = len(self.y_index[tmp_y])
            
        for tmp_yz in self.yz_tuple:
            self.yz_len[tmp_yz] = len(self.yz_index[tmp_yz])

        # Default batch size
        self.S = {}
        
        for tmp_yz in self.yz_tuple:
            self.S[tmp_yz] = self.batch_size * (self.yz_len[tmp_yz])/self.N

        
        self.lb1 = (self.S[1,1])/(self.S[1,1]+(self.S[1,0]))
        self.lb2 = (self.S[-1,1])/(self.S[-1,1]+(self.S[-1,0]))

  
    def adjust_lambda(self):
        """Adjusts the lambda values for FairBatch algorithm.
        
        The detailed algorithms are decribed in the paper.

        """
        
        self.model.eval()
        logit = self.model(self.x_data).squeeze()

        criterion = torch.nn.BCELoss(reduction = 'none')
                    
        if self.fairness_type == 'eqopp':
            
            yhat_yz = {}
            yhat_y = {}
                        
            eo_loss = criterion ((F.tanh(logit)+1)/2, (self.y_data+1)/2)
            
            for tmp_yz in self.yz_tuple:
                yhat_yz[tmp_yz] = float(torch.sum(eo_loss[self.yz_index[tmp_yz]])) / self.yz_len[tmp_yz]
                
            for tmp_y in self.y_item:
                yhat_y[tmp_y] = float(torch.sum(eo_loss[self.y_index[tmp_y]])) / self.y_len[tmp_y]
            
            # lb1 * loss_z1 + (1-lb1) * loss_z0
            
            if yhat_yz[(1, 1)] > yhat_yz[(1, 0)]:
                self.lb1 += self.alpha
            else:
                self.lb1 -= self.alpha
                
            if self.lb1 < 0:
                self.lb1 = 0
            elif self.lb1 > 1:
                self.lb1 = 1 
                
        elif self.fairness_type == 'eqodds':
            
            yhat_yz = {}
            yhat_y = {}
                        
            eo_loss = criterion ((F.tanh(logit)+1)/2, (self.y_data+1)/2)
            
            for tmp_yz in self.yz_tuple:
                yhat_yz[tmp_yz] = float(torch.sum(eo_loss[self.yz_index[tmp_yz]])) / self.yz_len[tmp_yz]
                
            for tmp_y in self.y_item:
                yhat_y[tmp_y] = float(torch.sum(eo_loss[self.y_index[tmp_y]])) / self.y_len[tmp_y]
            
            y1_diff = abs(yhat_yz[(1, 1)] - yhat_yz[(1, 0)])
            y0_diff = abs(yhat_yz[(-1, 1)] - yhat_yz[(-1, 0)])
            
            # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
            # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0
            
            if y1_diff > y0_diff:
                if yhat_yz[(1, 1)] > yhat_yz[(1, 0)]:
                    self.lb1 += self.alpha
                else:
                    self.lb1 -= self.alpha
            else:
                if yhat_yz[(-1, 1)] > yhat_yz[(-1, 0)]:
                    self.lb2 += self.alpha
                else:
                    self.lb2 -= self.alpha
                    
                
            if self.lb1 < 0:
                self.lb1 = 0
            elif self.lb1 > 1:
                self.lb1 = 1
                
            if self.lb2 < 0:
                self.lb2 = 0
            elif self.lb2 > 1:
                self.lb2 = 1
                
        elif self.fairness_type == 'dp':
            yhat_yz = {}
            yhat_y = {}
            
            ones_array = np.ones(len(self.y_data))
            ones_tensor = torch.FloatTensor(ones_array)
            
            if self.useCuda:
                ones_tensor = ones_tensor.cuda()
                
            dp_loss = criterion((F.tanh(logit)+1)/2, ones_tensor) # Note that ones tensor puts as the true label
                       
            for tmp_yz in self.yz_tuple:
                yhat_yz[tmp_yz] = float(torch.sum(dp_loss[self.yz_index[tmp_yz]])) / self.z_len[tmp_yz[1]]
                    
            
            y1_diff = abs(yhat_yz[(1, 1)] - yhat_yz[(1, 0)])
            y0_diff = abs(yhat_yz[(-1, 1)] - yhat_yz[(-1, 0)])
            
            # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
            # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0
            
            if y1_diff > y0_diff:
                if yhat_yz[(1, 1)] > yhat_yz[(1, 0)]:
                    self.lb1 += self.alpha
                else:
                    self.lb1 -= self.alpha
            else:
                if yhat_yz[(-1, 1)] > yhat_yz[(-1, 0)]: 
                    self.lb2 -= self.alpha
                else:
                    self.lb2 += self.alpha
                    
            if self.lb1 < 0:
                self.lb1 = 0
            elif self.lb1 > 1:
                self.lb1 = 1
                
            if self.lb2 < 0:
                self.lb2 = 0
            elif self.lb2 > 1:
                self.lb2 = 1
                
        elif self.fairness_type == 'dpeo':
            
            # For DP
            dp_yhat_yz = {}
            dp_yhat_y = {}
            
            ones_array = np.ones(len(self.y_data))
            ones_tensor = torch.FloatTensor(ones_array)
            
            if self.useCuda:
                ones_tensor = ones_tensor.cuda()
                
            dp_loss = criterion((F.tanh(logit)+1)/2, ones_tensor) # Note that ones tensor puts as the true label
            
            for tmp_yz in self.yz_tuple:
                dp_yhat_yz[tmp_yz] = float(torch.sum(dp_loss[self.yz_index[tmp_yz]])) / self.z_len[tmp_yz[1]]
                    
            
            dp_y1_diff = abs(dp_yhat_yz[(1, 1)] - dp_yhat_yz[(1, 0)])
            dp_y0_diff = abs(dp_yhat_yz[(-1, 1)] - dp_yhat_yz[(-1, 0)])
            
            # For EO
            eo_yhat_yz = {}
            eo_yhat_y = {}
            
            eo_loss = criterion ((F.tanh(logit)+1)/2, (self.y_data+1)/2)
            
            for tmp_yz in self.yz_tuple:
                eo_yhat_yz[tmp_yz] = float(torch.sum(eo_loss[self.yz_index[tmp_yz]])) / self.yz_len[tmp_yz]
                
            for tmp_y in self.y_item:
                eo_yhat_y[tmp_y] = float(torch.sum(eo_loss[self.y_index[tmp_y]])) / self.y_len[tmp_y]
            
            # Apply the tuning knob to adjust the importance among DP and EO
            eo_y1_diff = self.knob * abs(eo_yhat_yz[(1, 1)] - eo_yhat_yz[(1, 0)])
            eo_y0_diff = self.knob * abs(eo_yhat_yz[(-1, 1)] - eo_yhat_yz[(-1, 0)])
            
            max_diff = max(dp_y1_diff, dp_y0_diff, eo_y1_diff, eo_y0_diff)
            
            
            if max_diff == dp_y1_diff:
                if dp_yhat_yz[(1, 1)] > dp_yhat_yz[(1, 0)]:
                    self.lb1 += self.alpha
                else:
                    self.lb1 -= self.alpha
            elif max_diff == dp_y0_diff:
                if dp_yhat_yz[(-1, 1)] > dp_yhat_yz[(-1, 0)]: 
                    self.lb2 -= self.alpha
                else:
                    self.lb2 += self.alpha
            elif max_diff == eo_y1_diff:
                if eo_yhat_yz[(1, 1)] > eo_yhat_yz[(1, 0)]:
                    self.lb1 += self.alpha
                else:
                    self.lb1 -= self.alpha
            else:
                if eo_yhat_yz[(-1, 1)] > eo_yhat_yz[(-1, 0)]:
                    self.lb2 += self.alpha
                else:
                    self.lb2 -= self.alpha
            
            if self.lb1 < 0:
                self.lb1 = 0
            elif self.lb1 > 1:
                self.lb1 = 1
                
            if self.lb2 < 0:
                self.lb2 = 0
            elif self.lb2 > 1:
                self.lb2 = 1
            
    
    def select_batch_replacement(self, batch_size, full_index, batch_num, replacement = False):
        """Selects a certain number of batches based on the given batch size.
        
        Args: 
            batch_size: An integer for the data size in a batch.
            full_index: An array containing the candidate data indices.
            batch_num: An integer indicating the number of batches.
            replacement: A boolean indicating whether a batch consists of data with or without replacement.
        
        Returns:
            Indices that indicate the data.
            
        """
        
        select_index = []
        
        if replacement == True:
            for _ in range(batch_num):
                select_index.append(np.random.choice(full_index, batch_size, replace = False))
        else:
            tmp_index = full_index.detach().cpu().numpy().copy()
            random.shuffle(tmp_index)
            
            start_idx = 0
            for i in range(batch_num):
                if start_idx + batch_size > len(full_index):
                    select_index.append(np.concatenate((tmp_index[start_idx:], tmp_index[ : batch_size - (len(full_index)-start_idx)])))
                    
                    start_idx = len(full_index)-start_idx
                else:

                    select_index.append(tmp_index[start_idx:start_idx + batch_size])
                    start_idx += batch_size
            
        return select_index

    def __iter__(self):
        """Iters the full process of FairBatch for serving the batches to training.
        
        Returns:
            Indices that indicate the data in each batch.
            
        """
              
        if self.fairness_type == 'original':
            
            entire_index = torch.FloatTensor([i for i in range(len(self.y_data))])
            
            sort_index = self.select_batch_replacement(self.batch_size, entire_index, self.batch_num, self.replacement)
            
            for i in range(self.batch_num):
                yield sort_index[i]
            
        else:
            
            if self.fairness_type == 'grid':
                each_size = {}
                each_size[(1,1)] = self.grid[(1,1)]
                each_size[(1,0)] = self.grid[(1,1)]
                each_size[(-1,1)] = self.grid[(-1,1)]
                each_size[(-1,0)] = self.grid[(-1,0)]
                
            else:    
                self.adjust_lambda() # Adjust the lambda values
                each_size = {}
                        
                # Based on the updated lambdas, determine the size of each class in a batch
                if self.fairness_type == 'eqopp':
                    # lb1 * loss_z1 + (1-lb1) * loss_z0

                    each_size[(1,1)] = round(self.lb1 * (self.S[(1,1)] + self.S[(1,0)]))
                    each_size[(1,0)] = round((1-self.lb1) * (self.S[(1,1)] + self.S[(1,0)]))
                    each_size[(-1,1)] = round(self.S[(-1,1)])
                    each_size[(-1,0)] = round(self.S[(-1,0)])

                elif self.fairness_type == 'eqodds':
                    # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
                    # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0

                    each_size[(1,1)] = round(self.lb1 * (self.S[(1,1)] + self.S[(1,0)]))
                    each_size[(1,0)] = round((1-self.lb1) * (self.S[(1,1)] + self.S[(1,0)]))
                    each_size[(-1,1)] = round(self.lb2 * (self.S[(-1,1)] + self.S[(-1,0)]))
                    each_size[(-1,0)] = round((1-self.lb2) * (self.S[(-1,1)] + self.S[(-1,0)]))
                    

                elif self.fairness_type == 'dp':
                    # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
                    # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0

                    each_size[(1,1)] = round(self.lb1 * (self.S[(1,1)] + self.S[(1,0)]))
                    each_size[(1,0)] = round((1-self.lb1) * (self.S[(1,1)] + self.S[(1,0)]))
                    each_size[(-1,1)] = round(self.lb2 * (self.S[(-1,1)] + self.S[(-1,0)]))
                    each_size[(-1,0)] = round((1-self.lb2) * (self.S[(-1,1)] + self.S[(-1,0)]))
                    
                elif self.fairness_type == 'dpeo':
                    # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
                    # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0

                    each_size[(1,1)] = round(self.lb1 * (self.S[(1,1)] + self.S[(1,0)]))
                    each_size[(1,0)] = round((1-self.lb1) * (self.S[(1,1)] + self.S[(1,0)]))
                    each_size[(-1,1)] = round(self.lb2 * (self.S[(-1,1)] + self.S[(-1,0)]))
                    each_size[(-1,0)] = round((1-self.lb2) * (self.S[(-1,1)] + self.S[(-1,0)]))


            # Get the indices for each class
            sort_index_y_1_z_1 = self.select_batch_replacement(each_size[(1, 1)], self.yz_index[(1,1)], self.batch_num, self.replacement)
            sort_index_y_0_z_1 = self.select_batch_replacement(each_size[(-1, 1)], self.yz_index[(-1,1)], self.batch_num, self.replacement)
            sort_index_y_1_z_0 = self.select_batch_replacement(each_size[(1, 0)], self.yz_index[(1,0)], self.batch_num, self.replacement)
            sort_index_y_0_z_0 = self.select_batch_replacement(each_size[(-1, 0)], self.yz_index[(-1,0)], self.batch_num, self.replacement)
            
                
            for i in range(self.batch_num):
                key_in_fairbatch = sort_index_y_0_z_0[i].copy()
                key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_y_1_z_0[i].copy()))
                key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_y_0_z_1[i].copy()))
                key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_y_1_z_1[i].copy()))
                             
                random.shuffle(key_in_fairbatch)

                yield key_in_fairbatch
                               

    def __len__(self):
        """Returns the length of data."""
        
        return len(self.y_data)
