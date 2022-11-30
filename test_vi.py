import os


from torch import utils, Generator
import multiprocessing
import numpy as np

from dataloader import MoNADataset
from mean_field_vi import BayesianNetwork

targ_v = 1
log_dir = 'logs/mfvi'

dataset = MoNADataset(fingerprint_type='MACCS')
train_set, test_set, extra = utils.data.random_split(dataset, [0.8, 0.2, 0], generator=Generator().manual_seed(2022))

print(f"INPUT_SIZE: {len(train_set[0][0])}")

model = BayesianNetwork.load_from_checkpoint(f"{log_dir}/best.ckpt", lr=1e-3, hidden_layer_sizes=[256],
                                             input_size=len(train_set[0][0]), output_size=1000, samples=8)

num_workers = multiprocessing.cpu_count()
TEST_BATCH_SIZE = 1024

test_loader = utils.data.DataLoader(test_set, num_workers=num_workers, batch_size=TEST_BATCH_SIZE, drop_last=True) 

def get_calibration(conf, posterior_pred, y, add=0.0):
        # First find confidence interval bounds
        lower_bound = np.percentile(posterior_pred, (0.5-(conf/2.0))*100, axis=1)
        upper_bound = np.percentile(posterior_pred, (0.5+(conf/2.0))*100, axis=1) + add

        print(posterior_pred.shape)
        
        # Find confidence interval containment by logical AND on upper and lower bounds surrounding the true value
        lower_true = np.less_equal(lower_bound, y)
        print(f"bounded_below by {(0.5-(conf/2.0)):0.2f} is {np.sum(lower_true):0.2f}")
        upper_true = np.greater_equal(upper_bound, y)
        print(f"bounded_above by {(0.5+(conf/2.0)):0.2f} is {np.sum(upper_true):0.2f}")
        contained = np.logical_and(lower_true, upper_true)
        
        calibration = np.mean(contained)
        print(f"contained in {conf:0.2f}-interval is {np.sum(contained):0.2f}")
        print(f"Calibration (interval) at {conf:0.2f} is {calibration:0.2f}")
        return calibration
    

y = None
n_samples = 16
# Monte Carlo Posterior Predictive
tracked = []
peaks = []
for (x, y, *_) in test_loader:
    
    y = y.detach().numpy()
    batch_size = y.shape[0]
    posterior_pred = list() 
    
    # Select highest peaks
    indices = np.argmax(y, axis=1)
    peaks.append(indices)
    
    for i in range(n_samples):
         
        y_hat = model.forward(x, sample=True).detach().numpy()
        
        _posterior_sample = np.exp(y_hat)
            
        posterior_pred.append(_posterior_sample)

    posterior_pred = np.stack(posterior_pred, axis=2)
   
    # Predictive Posterior Estimate 
    ppe = np.mean(posterior_pred, axis=2)

    bounds = []
    for conf in [0.5]:
        # First find confidence interval bounds
        lower_bound = np.percentile(posterior_pred, (0.5-(conf/2.0))*100, axis=2)
        upper_bound = np.percentile(posterior_pred, (0.5+(conf/2.0))*100, axis=2)
        bounds.append((lower_bound, upper_bound))

    tracked.append((ppe, bounds))
    
peaks2 = np.stack(peaks, axis=1).reshape(-1)

ppe = np.stack([i[0] for i in tracked], axis=0).reshape(-1, 1000)
bounds = list()
for i in range(len(tracked[0][1])):
    print(np.stack([j[1][i][0] for j in tracked], axis=0).shape)
    l_bounds = np.stack([j[1][i][0] for j in tracked], axis=0).reshape(-1,1000)
    u_bounds = np.stack([j[1][i][1] for j in tracked], axis=0).reshape(-1,1000)
    bounds.append((l_bounds, u_bounds))

def get_rank(scores, peaks2)