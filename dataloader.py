from torch.utils.data import Dataset
import torch
import numpy as np


import pickle as pkl
import os


"""
    Constructor: (data_dir, fingerprint filename, mass spec file name)
"""
class MoNADataset(Dataset):
    def __init__(self, data_dir = './Data', fps_name = 'fps_maccs', ms_name='msms'):
        super(Dataset).__init__()

        if '.pkl' not in fps_name:
            fps_name = fps_name + '.pkl'
        if '.pkl' not in ms_name:
            ms_name = ms_name + '.pkl'

        self.fps = None
        with open(os.path.join(data_dir, fps_name), 'rb') as f:
            self.fps = pkl.load(f)
        self.fps = [torch.Tensor(i) for i in self.fps]
        
        self._len = len(self.fps)
        
        self.msms = None
        with open(os.path.join(data_dir, ms_name), 'rb') as f:
            self.msms = pkl.load(f)
        
        """
            Normalization of Mass Spectra
            1. The largest peak is treated as intensity '1.0', all other rescaled.
            2. All peaks with magnitude < 0.01 are masked as 0.

        """
        self.msms = [np.array(i) for i in self.msms]
        """self.msms = [i/np.max(i) for i in self.msms]
        thr = 0.00
        for i in range(len(self)):
            self.msms[i][self.msms[i] < thr] = 0;
        """ 
        self.msms = [torch.Tensor(i) for i in self.msms]

        nonzero_weight = 0.9 ; zero_weight = 0.1;

        self.masks = []
        for i in range(len(self)):
            nonzero_mask = self.msms[i] != 0
            zero_mask = ~nonzero_mask

            nonzero_mask = nonzero_mask / (nonzero_weight * nonzero_mask.sum());
            zero_mask = zero_mask / (zero_weight * zero_mask.sum());
            
            mask = zero_mask + nonzero_mask
            self.masks.append(mask)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        x = self.fps[idx]
        y = self.msms[idx]
        mask = self.masks[idx]
        return (x, y, mask)


if __name__ == '__main__':
    dataset = MoNADataset();

    for x, y in dataset:
        print(f"training data shape is {x.shape}")
        print(f"Mass spec label shape is {y.shape}")
        break;
