import re
import pickle as pkl
import os
import json
import contextlib
from io import StringIO
import warnings


import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from scipy.stats import binned_statistic
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect


def process_msms(msms: str):

    mz_int_list = msms.split(' ')
    mz_array = np.zeros(len(mz_int_list))
    intensity_array = np.zeros(len(mz_int_list))

    for i, mz_int in enumerate(mz_int_list):
        mz, intensity = mz_int.split(':')
        mz_array[i] = float(mz)
        intensity_array[i] = float(intensity)

    return mz_array, intensity_array


def bin_msms(mz_array, intensity_array, mass_range=[0,1000], bin_width=1):

    bins = np.arange(mass_range[0], mass_range[1] + bin_width, bin_width)

    statistic, _, _, = binned_statistic(mz_array, intensity_array, bins=bins)

    msms_vec = [x if not np.isnan(x) else 0 for x in statistic]

    return msms_vec


def generate_fingerprint(smiles, type='ECFP'):
    mol = MolFromSmiles(smiles)

    fingerprint = None
    if type.upper() == 'MACCS':
        fingerprint = MACCSkeys.GenMACCSKeys(mol)
    # By default, ecfp-4 aka diameter 4 or radius 2
    if type.upper() == 'ECFP':
        fingerprint = GetMorganFingerprintAsBitVect(mol, 2, nBits=512)

    return fingerprint.ToList()

def preprocess_data(data_dir, type='ECFP'):
    
    data = None
    with open(os.path.join(data_dir, 'MoNA-export-LC-MS-MS_Spectra.json')) as f:
        data = json.load(f)

    print("Total number of spectra: ", len(data))

    all_msms = []
    all_fps = []
    failed_fps = []
    error_msgs = StringIO()


    for idx in tqdm(reversed(range(len(data))), desc=f'{type} fingerprinting', total=len(data)):

        x = data[idx]

        msms = x['spectrum']
        mz_array, intensity_array = process_msms(msms)

        # skip msms with mz > 1000
        if max(mz_array) > 1000: 
            continue

        msms_vec = bin_msms(mz_array, intensity_array)

        # Hide Rdkit error messages4
        with contextlib.redirect_stderr(error_msgs):
            
            try:
                for mD in x['compound'][0]['metaData']:
                    if mD['name'] == 'SMILES':
                        smiles = mD['value']
                        break
                fp = generate_fingerprint(smiles, type=type)
            except:
                failed_fps.append(idx)
                continue

        all_msms.append(msms_vec)
        all_fps.append(fp)
        
        del data[idx] # Reduces memory consumption, as it rises with the creation of fingerprint objects

    print("Number of msms and fps processed: ", len(all_fps))
    print("Number of failed instances: ", len(failed_fps))
    error_msgs = error_msgs.getvalue();

    all_fps = np.array(all_fps)
    with open(os.path.join(data_dir, f"msms_{type.lower()}.pkl"), 'wb') as f:
        pkl.dump(all_msms, f)
    with open(os.path.join(data_dir, f"fps_{type.lower()}.pkl"), 'wb') as f:
        pkl.dump(all_fps, f)
    with open(os.path.join(data_dir, f"errors_{type.lower()}.pkl"), 'wb') as f:
        pkl.dump(error_msgs, f)


"""
    Constructor: (data_dir, fingerprint filename, mass spec file name)
"""
class MoNADataset(Dataset):
    def __init__(self, data_dir = './data', fingerprint_type='ECFP', force=False):
        super(Dataset).__init__()
        
        
        assert fingerprint_type in ['MACCS', 'ECFP'], 'Invalid Fingerprint Type';
        fps_name = f'fps_{fingerprint_type.lower()}'; ms_name=f'msms_{fingerprint_type.lower()}';

        if '.pkl' not in fps_name:
            fps_name = fps_name + '.pkl'
        if '.pkl' not in ms_name:
            ms_name = ms_name + '.pkl'

        if force or not os.path.exists(os.path.join(data_dir, fps_name)) or not os.path.exists(os.path.join(data_dir, ms_name)):

            print("Generating Fingerprints...")
            if not os.path.exists(os.path.join(data_dir, 'MoNA-export-LC-MS-MS_Spectra.json')):
                print("Downloading dataset...")
                if not os.path.exists(data_dir):
                    os.system(f'mkdir -p {data_dir}')
                os.system(f"wget https://mona.fiehnlab.ucdavis.edu/rest/downloads/retrieve/9c822c48-67f4-4600-8b81-ef7491008245")
                os.system(f"unzip 9c822c48-67f4-4600-8b81-ef7491008245")
                os.system(f"mv MoNA-export-LC-MS-MS_Spectra.json {data_dir}")
                os.system(f"rm 9c822c48-67f4-4600-8b81-ef7491008245")

            preprocess_data(data_dir, type=fingerprint_type)


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
        """
        self.msms = [i/np.max(i) for i in self.msms]
        thr = 0.00
        for i in range(len(self)):
            self.msms[i][self.msms[i] < thr] = 0;
        """ 
        self.msms = [torch.Tensor(i) for i in self.msms]

        """
            Add sparse/nonsparse weights
        """
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
    dataset = MoNADataset(force=True);

    for x, y, *r in dataset:
        print(f"training data shape is {x.shape}")
        print(f"Mass spec label shape is {y.shape}")
        break;
