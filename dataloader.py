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
from rdkit import RDLogger
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

N_MORGAN_BITS=1024

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
    elif type.upper() == 'ECFP':
        fingerprint = GetMorganFingerprintAsBitVect(mol, 2, nBits=N_MORGAN_BITS)
    elif type.upper() == 'SMILES':
        fingerprint = smiles

    return fingerprint.ToList()

def preprocess_data(data_dir, type='ECFP', normalize_peaks=False, zero_threshold=0):
    
    data = None
    with open(os.path.join(data_dir, '..', 'MoNA-export-LC-MS-MS_Spectra.json')) as f:
        data = json.load(f)

    print("Total number of spectra: ", len(data))

    all_msms = []
    all_fps = []
    failed_fps = []
    error_msgs = StringIO()

    RDLogger.DisableLog('rdApp.*')     

    for idx in tqdm(reversed(range(len(data))), desc=f'{type} fingerprinting', total=len(data)):

        x = data[idx]

        msms = x['spectrum']
        mz_array, intensity_array = process_msms(msms)

        # skip msms with mz > 1000
        if max(mz_array) > 1000: 
            continue

        msms_vec = bin_msms(mz_array, intensity_array)

        # Hide Rdkit error messages4
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

    RDLogger.EnableLog('rdApp.*')     
    print("Number of msms and fps processed: ", len(all_fps))
    print("Number of failed instances: ", len(failed_fps))
    error_msgs = error_msgs.getvalue();
        
    all_msms = [np.array(i) for i in all_msms]
    
    """
        Normalization of Mass Spectra
        1. The largest peak is treated as intensity '1.0', all other rescaled.
        2. All peaks with magnitude < 0.01 are masked as 0.

    """
    if normalize_peaks:
        all_msms = [i/np.max(i) for i in all_msms]
        if zero_threshold > 0:
            for i in range(len(all_msms)):
                all_msms[i][all_msms[i] < zero_threshold] = 0;
    all_msms = [torch.Tensor(i) for i in all_msms]

    all_fps = [torch.Tensor(i) for i in np.array(all_fps)]
    with open(os.path.join(data_dir, f"msms_{type.lower()}.pkl"), 'wb') as f:
        pkl.dump(all_msms, f)
    with open(os.path.join(data_dir, f"fps_{type.lower()}.pkl"), 'wb') as f:
        pkl.dump(all_fps, f)
    with open(os.path.join(data_dir, f"errors_{type.lower()}.pkl"), 'wb') as f:
        pkl.dump(error_msgs, f)

    return all_fps, all_msms

        
def preprocess_mask(msms, sparse_weight, filename, bayesian_mask=False):
    """
        Add sparse/nonsparse weights
    """
    zero_weight = sparse_weight
    nonzero_weight = 1 - sparse_weight
    
    masks = []
    for i in range(len(msms)):
        nonzero_mask = msms[i] != 0
        zero_mask = ~nonzero_mask


        nonzero_uniform_weight = nonzero_weight/nonzero_mask.sum();
        zero_uniform_weight = zero_weight/zero_mask.sum();

        """
        We choose a sigma yielding a log_prob data likelihood of nonzero_uniform_weight * (y_hat - y)^2 up to constant
        nonzero_uniform_weight = 1 / (2*((mask*tau) ** 2))
        We will use tau as absolute scale on the log_likelihood term, for now it is 1.
        nonzero_uniform_weight = 1 / (2*((mask) ** 2))
        mask = \sqrt{1/(2*nonzero_uniform_weight)}
        """
        if bayesian_mask:
            nonzero_uniform_bayesian_sigma = np.sqrt(1.0/(2*nonzero_uniform_weight))
            nonzero_mask = nonzero_uniform_bayesian_sigma*nonzero_mask
            
            zero_uniform_bayesian_sigma = np.sqrt(1.0/(2*zero_uniform_weight))
            zero_mask = zero_uniform_bayesian_sigma*zero_mask
        else:
            nonzero_mask = nonzero_uniform_weight*nonzero_mask
            zero_mask = zero_uniform_weight*zero_mask
            
        
        mask = zero_mask + nonzero_mask
        masks.append(mask)
    
    with open(filename, 'wb') as f:
        pkl.dump(masks, f)
    
    return masks


class MoNADataset(Dataset):
    """
    normalize_peaks: bool
        if True, normalizes intensities vector by using linear max-normalization, dividing all peak intensities by the max peak
        
    zero_threshold: bool
        if normalize_peaks is True, all resultant normalized peaks with values below this threshold are set to 0, (aka treated as noise) 

    sparse_weight: float
        in range [0,1.0], represents percentage of probability mass assigned to 0-entries for regression.
    """
    def __init__(self, data_dir = './data', fingerprint_type='ECFP', normalize_peaks=False, zero_threshold=0,
                 bayesian_mask=False, sparse_weight=0.5, force=False):
        super(Dataset).__init__()
        
        self.data_dir = data_dir
        cache_dir = os.path.join(data_dir, 'mona_cache')
        self.cache_dir = cache_dir
        
        assert fingerprint_type in ['MACCS', 'ECFP', 'SMILES'], 'Invalid Fingerprint Type';
        fps_name = f'fps_{fingerprint_type.lower()}'; ms_name=f'msms_{fingerprint_type.lower()}';

        if '.pkl' not in fps_name:
            fps_name = fps_name + '.pkl'
        if '.pkl' not in ms_name:
            ms_name = ms_name + '.pkl'
        
        mask_name = f"{fingerprint_type.lower()}_sparse={sparse_weight:.2f}_normalized={int(normalize_peaks)}_mask"

        if bayesian_mask:
            mask_name = f"{mask_name}_bayesian.pkl"
        else:
            mask_name = f"{mask_name}.pkl"

        self.fps, self.msms = None, None
        if force or not os.path.exists(os.path.join(cache_dir, fps_name)) or not os.path.exists(os.path.join(cache_dir, ms_name)):
            
            if force:
                print("Force ", end='')
            print(f"Generating {fingerprint_type} Fingerprints...")
                
            if not os.path.exists(cache_dir):
                os.system(f'mkdir -p {cache_dir}')
            
            if not os.path.exists(os.path.join(data_dir, 'MoNA-export-LC-MS-MS_Spectra.json')):
                print("Downloading dataset...")
                os.system(f"wget https://mona.fiehnlab.ucdavis.edu/rest/downloads/retrieve/9c822c48-67f4-4600-8b81-ef7491008245")
                os.system(f"unzip 9c822c48-67f4-4600-8b81-ef7491008245")
                os.system(f"mv MoNA-export-LC-MS-MS_Spectra.json {data_dir}")
                os.system(f"rm 9c822c48-67f4-4600-8b81-ef7491008245")

            self.fps, self.msms = preprocess_data(cache_dir, type=fingerprint_type,
                                                  normalize_peaks=normalize_peaks, zero_threshold=zero_threshold)

        else:
            
            with open(os.path.join(cache_dir, fps_name), 'rb') as f:
                self.fps = pkl.load(f)
            with open(os.path.join(cache_dir, ms_name), 'rb') as f:
                self.msms = pkl.load(f) 

        self._len = len(self.fps)
            
        self.masks = None 
        if force or not os.path.exists(os.path.join(cache_dir, mask_name)):
            if force:
                print("Force ", end='')
            if bayesian_mask:
                print("Generating Bayesian Loss Weighing Masks")
            else:
                print("Generating Loss Weighing Masks")
            self.masks = preprocess_mask(msms=self.msms, sparse_weight=sparse_weight,
                                         bayesian_mask=bayesian_mask, filename=os.path.join(cache_dir, mask_name))
        else:
            with open(os.path.join(cache_dir, mask_name), 'rb') as f:
                self.masks = pkl.load(f)
            

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        x = self.fps[idx]
        y = self.msms[idx]
        mask = self.masks[idx]
        return (x, y, mask)


if __name__ == '__main__':
    dataset = MoNADataset(fingerprint_type='MACCS',
                          sparse_weight=0.9,
                          force=True);

    for x, y, *r in dataset:
        print(f"training data shape is {x.shape}")
        print(f"Mass spec label shape is {y.shape}")
        break;
