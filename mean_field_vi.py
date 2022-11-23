import os

import numpy as np
import math
import torch

from torch import optim, nn, utils, Tensor, Generator
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import pytorch_lightning as pl

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


class BayesianNetwork(pl.LightningModule):
    def __init__(self, 
                 lr,
                 INPUT_SIZE=167, 
                 OUTPUT_SIZE=1000, 
                 q_sigma=0.2,
                 hidden_layer_sizes=[4096,4096,4096,4096,4096],
                 samples=2):
      
        super().__init__()
        self.lr = lr
        self.q_sigma = torch.Tensor([float(q_sigma)])
        self.samples = samples
        self.output_size = OUTPUT_SIZE
        self.input_size = INPUT_SIZE

        nn_layer_sizes = ([INPUT_SIZE] + hidden_layer_sizes + [OUTPUT_SIZE])
        self.n_layers = len(nn_layer_sizes) - 1

        # create neural network, layer by layer
        self.layer_activations = list()
        self.layer_params = nn.ModuleList()
        self.ffn = nn.Sequential()
        for layer_id, (n_in, n_out) in enumerate(zip(nn_layer_sizes[:-1], nn_layer_sizes[1:])):
            self.ffn.add_module("layer" + str(layer_id), nn.Linear(n_in, n_out))
            if layer_id < self.n_layers:
              self.ffn.add_module("layer" + str(layer_id) + "_relu", nn.ReLU())
            else:
              self.ffn.add_module("layer" + str(layer_id) + "_softmax", nn.Softmax())

        # Prior distributions
        self.weight_prior = Gaussian(0,q_sigma)
        self.bias_prior = Gaussian(0,q_sigma)

        self.likelihood_weight = []
        self.likelihood_bias = []
        
        for name, ll in self.ffn.named_children():
          if isinstance(ll, nn.Linear):
            weight = ll.weight
            bias = ll.bias
            self.likelihood_weight.append(Gaussian(weight, torch.randn(weight.shape)))
            self.likelihood_bias.append(Gaussian(bias,torch.randn(bias.shape)))
        
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input, sample=False, calculate_log_probs=False):
        return self.ffn(input)

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward(retain_graph=True)

    def log_prior_and_posterior(self):
        for ll in range(self.n_layers):
          if self.training:
            weight = self.likelihood_weight[ll].sample()
            bias = self.likelihood_bias[ll].sample()
          else:
            weight = self.likelihood_weight[ll].mu
            bias = self.likelihood_bias[ll].mu
          if self.training:  
            self.log_prior+= self.weight_prior.log_prob(weight)
            self.log_prior+= self.bias_prior.log_prob(bias)
            self.log_variational_posterior+=self.likelihood_weight[ll].log_prob(weight)
            self.log_variational_posterior+=self.likelihood_bias[ll].log_prob(bias)
          else:
            self.log_prior, self.log_variational_posterior = 0, 0
        return self.log_prior, self.log_variational_posterior

    def sample_elbo(self, input, target, batch):
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
      loss = (log_variational_posterior - log_prior)/TRAIN_BATCH_SIZE + negative_log_likelihood
      return loss

    def training_step(self, batch, batch_idx):
        x, y, mask = batch

        loss = self.sample_elbo(x, y, batch)
        
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        y_hat = self.forward(x)
        loss = self.sample_elbo(x, y, batch)
    
        self.log("test_loss", loss)
        #self.log("test_argmax", y_hat[np.argmax(y)], prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer



if __name__ == '__main__':
    TRAIN_BATCH_SIZE = 1024
    TEST_BATCH_SIZE = 1024

    model = BayesianNetwork(lr=1e-3, hidden_layer_sizes=[4096])

    dataset = MoNADataset()
    train_set, test_set, extra = utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset)), 0], generator=Generator().manual_seed(2022))

    train_loader = utils.data.DataLoader(train_set, num_workers=12, batch_size=TRAIN_BATCH_SIZE) 
    test_loader = utils.data.DataLoader(test_set, batch_size=TEST_BATCH_SIZE) 

    trainer = pl.Trainer(max_epochs=50, auto_select_gpus = True, auto_scale_batch_size=True)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)