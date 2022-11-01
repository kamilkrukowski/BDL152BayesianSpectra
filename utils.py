
import numpy as np
from scipy.stats import binned_statistic

from rdkit.Chem import MolFromSmiles
from rdkit.Chem import MACCSkeys

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


def generate_fingerprint(smiles, type='MACCS'):
    mol = MolFromSmiles(smiles)
    if type == 'MACCS':
        fingerprint = MACCSkeys.GenMACCSKeys(mol)

    return fingerprint.ToList()