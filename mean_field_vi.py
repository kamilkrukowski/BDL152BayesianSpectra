import os

import numpy as np
import math
import torch

from torch import optim, nn, utils, Tensor, Generator
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import pytorch_lightning as pl
import multiprocessing
import sklearn.metrics

from dataloader import MoNADataset

class FixedSigmaGaussian(object):
    def __init__(self, mu, sigma):
        super().__init__()
        self.mu = mu
        self.normal = torch.distributions.Normal(0,1)
        self._sigma = sigma
    
    @property
    def sigma(self):
        return self._sigma
    
    def sample(self):
        epsilon = self.normal.sample(self.mu.size())
        return self.mu + self.sigma * epsilon
    
    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()

class FreeSigmaGaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0,1)
    
    @property
    def sigma(self):
        return torch.log1p(torch.exp(torch.as_tensor(self.rho)))
    
    def sample(self):
        epsilon = self.normal.sample(self.rho.size())
        return self.mu + self.sigma * epsilon
    
    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()
    
     
"""
    Torch.nn Layer implementation for Bayesian Mean Field Inference Layer
"""
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, q_sigma=0.2, fixed_sigma=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if fixed_sigma is not None and not torch.is_tensor(fixed_sigma):
            fixed_sigma = torch.tensor(fixed_sigma)

        if not torch.is_tensor(q_sigma):
            q_sigma = torch.tensor(q_sigma)

        # Weight parameters ; GLOROT initialization
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight_mu)

        self.weight_rho = None
        self.weight = None
        if fixed_sigma is None:
            self.weight_rho = nn.Parameter(torch.randn((out_features, in_features)))
            self.weight = FreeSigmaGaussian(self.weight_mu, self.weight_rho)
        else:
            self.weight = FixedSigmaGaussian(self.weight_mu, fixed_sigma)

        # Bias parameters ; GLOROT initialization
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        nn.init.normal_(self.bias_mu, std=0.5)
        
        self.bias_rho = None
        self.bias = None
        if fixed_sigma is None:
            self.bias_rho = nn.Parameter(torch.randn((out_features)))
            self.bias = FreeSigmaGaussian(self.bias_mu, self.bias_rho)
        else:
            self.bias = FixedSigmaGaussian(self.bias_mu, fixed_sigma)

        # Prior distribution
        self.weight_prior = FixedSigmaGaussian(0, q_sigma)
        self.bias_prior = FixedSigmaGaussian(0, q_sigma)

        self.log_prior = 0
        self.log_variational_posterior = 0
    
    def forward(self, input, sample=False):
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu

        if self.training:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0,0

        output = F.linear(input, weight, bias)
        return output
       
class BayesianNetwork(pl.LightningModule):
    def __init__(self, 
                lr,
                input_size, 
                output_size, 
                q_sigma=0.2,
                tau=0.2,
                hidden_layer_sizes=[10],
                samples=2,
                fixed_sigma=None):
        super().__init__()
        self.lr = lr
        self.samples = samples
        self.OUTPUT_SIZE = output_size
        self.INPUT_SIZE = input_size
        self.tau = tau

        if samples < 2:
            raise RuntimeError('Cannot have less than two monte carlo variational samples')

        nn_layer_sizes = ([input_size] + hidden_layer_sizes + [output_size])
        self.n_layers = len(nn_layer_sizes) - 1

        # create neural network, layer by layer
        self.layers = nn.ModuleList()
        for n_in, n_out in zip(nn_layer_sizes[:-1], nn_layer_sizes[1:]):
            layer = BayesianLinear(in_features=n_in, out_features=n_out, q_sigma=q_sigma, fixed_sigma=fixed_sigma)
            self.layers.append(layer)


    def forward(self, x, sample=False):
        for layer in self.layers:
            x = F.relu(layer(x, sample))
        return x
    
    def log_prior_and_posterior(self):
        log_prior = 0
        log_variational_posterior = 0
        for layer in self.layers:
           log_prior += layer.log_prior
           log_variational_posterior += layer.log_variational_posterior
        return log_prior, log_variational_posterior

    def sample_elbo(self, input, target, log_name=None, mask=None, prog_bar=False):
        
        """
            The likelihood function represents the likelihood associated with independent gaussian regression labels
            It is accurate to within a O(1) constant of the true likelihood (missing sqrt with pi)
        """
            
        samples = self.samples
        outputs = torch.zeros(samples, len(input), self.OUTPUT_SIZE)
        log_priors = torch.zeros(samples)
        log_variational_posteriors = torch.zeros(samples)
        for i in range(samples):
            outputs[i,:,:] = self.forward(input, sample=True)
            log_priors[i], log_variational_posteriors[i] = self.log_prior_and_posterior()
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        negative_log_likelihood = torch.mean(-torch.sum(((torch.square((torch.mean(outputs, dim=0) - target))) / (2 * (torch.square(mask*self.tau)))), dim=1))
        loss = (log_variational_posterior - log_prior) + negative_log_likelihood
        
        if log_name is not None:
            self.log(f'nll/{log_name}', negative_log_likelihood, prog_bar=prog_bar)
            self.log(f'kl/{log_name}', log_variational_posterior - log_prior, prog_bar=prog_bar)
        
        return loss

    def training_step(self, batch):
        x, y, mask = batch

        loss = self.sample_elbo(x, y, mask=mask, prog_bar=True)#, log_name='train') 
        
        self.log("loss/train", loss, on_epoch=True)
        return loss

    def get_metrics(self, y_hat, y, log_name, prog_bar=False):
        # Cosine similarity between (un)normalized peaks and model output 
        self.log(f"cosineSim/{log_name}",
                 F.cosine_similarity(y_hat, y).mean(),
                 prog_bar=prog_bar, on_epoch=True)
        # Mean AUROC of top-1 peak vs all other peaks across molecules
        self.log(f"mAUROC/{log_name}",
                 np.mean([sklearn.metrics.roc_auc_score(F.one_hot(np.argmax(y[i]), self.OUTPUT_SIZE).reshape(-1,1), y_hat[i]) for i in range(len(y))]),
                 prog_bar=prog_bar, on_epoch=True)
        # Mean top-1 peak Rank across molecules
        self.log(f"peakRank/{log_name}",
                 np.mean([float(i) for i in np.argmax(y_hat, axis=1)]),
                 prog_bar=prog_bar, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        y_hat = self.forward(x, sample=False)

        loss = self.sample_elbo(x, y, log_name='val', mask=mask)
        self.log("loss/val", loss)
        
        self.get_metrics(y_hat, y, log_name='val', prog_bar=True)        

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer



if __name__ == '__main__':
    TRAIN_BATCH_SIZE = 1024
    TEST_BATCH_SIZE = 1024
    EPOCHS = 50

    dataset = MoNADataset(fingerprint_type='MACCS', bayesian_mask=True, sparse_weight=0.1, normalize_peaks=False)
    
    train_set, test_set, extra = utils.data.random_split(dataset, [0.8, 0.2, 0], generator=Generator().manual_seed(2022))
    
    print(f"INPUT_SIZE: {len(train_set[0][0])}")

    num_workers = multiprocessing.cpu_count()
    train_loader = utils.data.DataLoader(train_set, num_workers=num_workers, batch_size=TRAIN_BATCH_SIZE) 
    test_loader = utils.data.DataLoader(test_set, num_workers=num_workers, batch_size=TEST_BATCH_SIZE) 

    METRIC = 'mAUROC/val'
    MODE = 'max'
    TRIAL_DIR = 'logs/mfvi2'
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        TRIAL_DIR, monitor=METRIC, mode=MODE, filename='best'
    )
    callbacks=[checkpoint_callback, pl.callbacks.ModelSummary(max_depth=-1), 
               pl.callbacks.EarlyStopping(monitor=METRIC, mode=MODE, patience=10, min_delta=0.001)]
    logger = pl.loggers.TensorBoardLogger(TRIAL_DIR)
    
    # Remove previous Tensorboard statistics
    if os.path.exists(TRIAL_DIR):
        os.system(f"rm -r {TRIAL_DIR}")
    os.system(f'mk -p {TRIAL_DIR}')

    model = BayesianNetwork(lr=1e-3, hidden_layer_sizes=[512], input_size=len(train_set[0][0]),
                            output_size=1000, samples=2, q_sigma=10.0, tau=0.01, fixed_sigma=0.1)

    trainer = pl.Trainer(max_epochs=EPOCHS, auto_select_gpus = True, auto_scale_batch_size=True,
                            callbacks=callbacks, logger=logger)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)