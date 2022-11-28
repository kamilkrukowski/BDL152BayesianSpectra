import pickle as pkl
import os


from torch_geometric.utils import from_smiles as graph_from_smiles
from tqdm.auto import tqdm


from dataloader import MoNADataset


class GNNDataset(MoNADataset):
    """
        normalize_peaks: bool
            if True, normalizes intensities vector by using linear max-normalization, dividing all peak intensities by the max peak
            
        zero_theshold: bool
            if normalize_peaks is True, all resultant normalized peaks with values below this threshold are set to 0, (aka treated as noise) 
    
    """
    def __init__(self, GNN, data_dir = './data', normalize_peaks=False, zero_threshold=1e-6, force=False):
        super(MoNADataset).__init__(data_dir=data_dir, fingerprint_type='GNN', normalize_peaks=normalize_peaks,
                                zero_threshold=zero_threshold, force=force)
        
        self.gnn = GNN
        
        fps_name = f"fps_pyg.pkl" 
        if force or not os.path.exists(os.path.join(data_dir, fps_name)):
            
            for idx in tqdm(reversed(range(len(self.fps))), desc=f"Processing Smiles into Graphs"):
                smiles = self.fps[idx]
                pyg_data = graph_from_smiles(smiles)
                self.fps[idx] = pyg_data
                
            with open(os.path.join(data_dir, fps_name), 'wb') as f:
                pkl.dump(self.fps, f)
        
        else:
            with open(os.path.join(data_dir, fps_name), 'rb') as f:
                self.fps = pkl.load(f)

    
    @property
    def training(self):
        return self.gnn.training
    
    @training.setter
    def _set_training(self, value: bool):
        assert value is bool, 'Error: training must be bool';
        self.gnn.train(value)

    def __getitem__(self, idx: int):
        x = self.fps[idx]
        y = self.msms[idx]
        mask = self.masks[idx]
        
        x = self.gnn(x);
        
        return (x, y, mask)

if __name__ == '__main__':
    dataset = GNNDataset(GNN=None, force=True);

    for x, y, *r in dataset:
        print(f"training data shape is {x.shape}")
        print(f"Mass spec label shape is {y.shape}")
        break;