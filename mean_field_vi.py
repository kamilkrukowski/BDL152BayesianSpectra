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


class Gaussian(object):
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
    
     
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, q_sigma=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.randn((out_features, in_features)))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)

        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.randn((out_features)))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)

        # Prior distribution
        self.weight_prior = Gaussian(0, q_sigma)
        self.bias_prior = Gaussian(0, q_sigma)
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
                 hidden_layer_sizes=[10],
                 samples=2):
      super().__init__()
      self.lr = lr
      self.samples = samples
      self.output_size = output_size
      self.input_size = input_size

      nn_layer_sizes = ([input_size] + hidden_layer_sizes + [output_size])
      self.n_layers = len(nn_layer_sizes) - 1

      # create neural network, layer by layer
      self.layers = nn.ModuleList()
      for n_in, n_out in zip(nn_layer_sizes[:-1], nn_layer_sizes[1:]):
          layer = BayesianLinear(in_features=n_in, out_features=n_out, q_sigma=q_sigma)
          self.layers.append(layer)

    def forward(self, x, sample=False):
        for layer_id, layer in enumerate(self.layers):
           if layer_id < self.n_layers:
              x = F.relu(layer(x, sample))
           else:
              x = F.log_softmax(layer(x, sample), dim=1)
        return x
    
    def log_prior_and_posterior(self):
        log_prior = 0
        log_variational_posterior = 0
        for layer in self.layers:
           log_prior += layer.log_prior
           log_variational_posterior += layer.log_variational_posterior
        return log_prior, log_variational_posterior

    def sample_elbo(self, input, target):
        samples = self.samples
        outputs = torch.zeros(samples, len(input), self.output_size)
        log_priors = torch.zeros(samples)
        log_variational_posteriors = torch.zeros(samples)
        for i in range(samples):
            outputs[i] = self.forward(input, sample=True)
            log_priors[i], log_variational_posteriors[i] = self.log_prior_and_posterior()
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        negative_log_likelihood = F.nll_loss(outputs.mean(0), torch.argmax(target, dim=1), reduction='sum')
        loss = (log_variational_posterior - log_prior) + negative_log_likelihood
        return loss

    def training_step(self, batch, batch_idx):
        x, y, *_ = batch

        loss = self.sample_elbo(x, y) 
        
        self.log("loss/train", loss)
        return loss

    def get_metrics(self, y_hat, y, log_name):
        # Cosine similarity between (un)normalized peaks and model output 
        self.log(f"cosineSim/{log_name}", F.cosine_similarity(y_hat, y).mean(), prog_bar=True)

        # Mean AUROC of top-1 peak vs all other peaks across molecules
        self.log(f"mAUROC/{log_name}", np.mean([sklearn.metrics.roc_auc_score(F.one_hot(np.argmax(y[i]), self.OUTPUT_SIZE).reshape(-1,1), y_hat[i]) for i in range(len(y))]), prog_bar=True)
        # Mean top-1 peak Rank across molecules
        self.log(f"peakRank/{log_name}", np.mean([float(i) for i in np.argmax(y_hat, axis=1)]), prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y, *_ = batch

        loss = self.sample_elbo(x, y)
        self.log("loss/val", loss)

        self.get_metrics(batch, 'val')        

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer



if __name__ == '__main__':
    TRAIN_BATCH_SIZE = 1024
    TEST_BATCH_SIZE = 1024
    EPOCHS = 50

    dataset = MoNADataset(fingerprint_type='MACCS')
    
    train_set, test_set, extra = utils.data.random_split(dataset, [0.8, 0.2, 0], generator=Generator().manual_seed(2022))
    
    print(f"INPUT_SIZE: {len(train_set[0][0])}")

    num_workers = multiprocessing.cpu_count()
    train_loader = utils.data.DataLoader(train_set, num_workers=num_workers, batch_size=TRAIN_BATCH_SIZE) 
    test_loader = utils.data.DataLoader(test_set, num_workers=num_workers, batch_size=TEST_BATCH_SIZE) 

    metric = 'mAUROC/val'
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        './lightning_logs', monitor=metric, mode='max'
    )
    callbacks=[checkpoint_callback, pl.callbacks.ModelSummary(max_depth=-1), 
               pl.callbacks.EarlyStopping(monitor=metric, mode="max", patience=5, min_delta=0.001)]

    model = BayesianNetwork(lr=1e-3, hidden_layer_sizes=[4096], input_size=len(train_set[0][0]), output_size=1000, samples=16)

    trainer = pl.Trainer(max_epochs=EPOCHS, auto_select_gpus = True, auto_scale_batch_size=True,
                            callbacks=callbacks)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)