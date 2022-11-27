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
                 num_batches,
                 input_size, 
                 output_size=1000, 
                 q_sigma=0.2,
                 hidden_layer_sizes=[4096,4096],
                 samples=2):
      super().__init__()
      self.lr = lr
      self.num_batches = num_batches
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

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward(retain_graph=True)

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
        negative_log_likelihood = F.nll_loss(outputs.mean(0), torch.argmax(target, dim=1), size_average=False)
        loss = (log_variational_posterior - log_prior)/self.num_batches + negative_log_likelihood
        return loss

    def training_step(self, batch, batch_idx):
        x, y, _ = batch

        loss = self.sample_elbo(x, y)
        
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        # y_hat = self.forward(x)
        loss = self.sample_elbo(x, y)
    
        self.log("test_loss", loss)
        #self.log("test_argmax", y_hat[np.argmax(y)], prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer



if __name__ == '__main__':
    TRAIN_BATCH_SIZE = 1024
    TEST_BATCH_SIZE = 1024

    dataset = MoNADataset()
    train_set, test_set, extra = utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset)), 0], generator=Generator().manual_seed(2022))

    num_workers = multiprocessing.cpu_count()
    train_loader = utils.data.DataLoader(train_set, num_workers=num_workers, batch_size=TRAIN_BATCH_SIZE) 
    test_loader = utils.data.DataLoader(test_set, batch_size=TEST_BATCH_SIZE) 

    model = BayesianNetwork(lr=1e-3, hidden_layer_sizes=[10], input_size=len(train_set[0][0]), num_batches=len(train_loader))

    trainer = pl.Trainer(max_epochs=10, auto_select_gpus = True, auto_scale_batch_size=True)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)