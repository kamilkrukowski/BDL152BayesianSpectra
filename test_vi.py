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
TEST_BATCH_SIZE=len(test_set)

test_loader = utils.data.DataLoader(test_set, num_workers=num_workers, batch_size=TEST_BATCH_SIZE) 

n_samples = 512
for (x, y, *_) in test_loader:
    y = y.detach().numpy()
    
    # Monte Carlo Posterior Predictive
    posterior_pred = []
    for i in range(n_samples):
        y_hat = model.forward(x, sample=True).detach().numpy()
        posterior_pred.append(y_hat)
    posterior_pred = np.stack(posterior_pred, axis=2)
    
    # We only consider calibration on highest peak
    posterior_pred = posterior_pred[np.argmax(y, axis=1)]
    
    for conf in [0.99]:
        # First find confidence interval bounds
        lower_bound = np.percentile(posterior_pred, (conf/2)*100, axis=2)
        upper_bound = np.percentile(posterior_pred, (1-conf/2)*100, axis=2)
        
        # Find confidence interval containment by logical AND on upper and lower bounds surrounding the true value
        lower_true = np.less_equal(lower_bound, y)
        print(f"lower_true at {conf:0.2f} is {np.sum(lower_true):0.2f}")
        upper_true = np.greater_equal(upper_bound, y)
        print(f"upper_true at {conf:0.2f} is {np.sum(upper_true):0.2f}")
        contained = np.logical_and(lower_true, upper_true)
        
        calibration = np.mean(contained)
        print(f"contained in {conf:0.2f} is {np.sum(contained):0.2f}")
        print(f"Calibration at {conf:0.2f} is {calibration:0.2f}")
        

            
    
    