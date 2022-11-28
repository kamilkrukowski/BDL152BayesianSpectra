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


class FFN(pl.LightningModule):
    def __init__(self, input_size, output_size, hidden_layer_sizes, lr=1.0e-3, weight_decay=None):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay

        self.INPUT_SIZE = input_size
        self.OUTPUT_SIZE = output_size

        hidden_layers = list()
        if len(hidden_layer_sizes) > 0:
            hidden_layers.append(nn.Linear(self.INPUT_SIZE, hidden_layer_sizes[0]))
            hidden_layers.append(nn.BatchNorm1d(hidden_layer_sizes[0]))
            hidden_layers.append(nn.Dropout(0.1))
        else:
            hidden_layers.append(nn.Linear(self.INPUT_SIZE, self.OUTPUT_SIZE))

        # Weird syntax
        for idx, _size in enumerate(hidden_layer_sizes[1:], start=1):
            hidden_layers.append(nn.Linear(hidden_layer_sizes[idx-1], hidden_layer_sizes[idx]))
            hidden_layers.append(nn.BatchNorm1d(hidden_layer_sizes[idx]))
            hidden_layers.append(nn.Dropout(0.5))
            
        hidden_layers.append(nn.Linear(hidden_layer_sizes[-1], self.OUTPUT_SIZE))
        
        self.ffn = nn.Sequential(*hidden_layers)


    def forward(self, x):
        return self.ffn(x)

    def get_loss(self, y_hat, y, mask=None):
        loss = 1 - F.cosine_similarity(y_hat, y, axis=1)
        return loss.mean();
    
    def get_metrics(self, y_hat, y, log_name):
        # Cosine similarity between (un)normalized peaks and model output 
        self.log(f"{log_name}_cosine_sim", F.cosine_similarity(y_hat, y).mean(), prog_bar=True)

        # Mean AUROC of top-1 peak vs all other peaks across molecules
        self.log(f"{log_name}_mAUROC", np.mean([sklearn.metrics.roc_auc_score(F.one_hot(np.argmax(y[i]), self.OUTPUT_SIZE).reshape(-1,1), y_hat[i]) for i in range(len(y))]), prog_bar=True)
        # Mean top-1 peak Rank across molecules
        self.log(f"{log_name}_peak_rank", np.mean([float(i) for i in np.argmax(y_hat, axis=1)]), prog_bar=True)

    def training_step(self, batch, batch_idx):
        x, y, *mask = batch
        y_hat = self.forward(x)
        loss = self.get_loss(y_hat, y)
        
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, *_ = batch
        y_hat = self.forward(x)
        loss = self.get_loss(y_hat, y)
        self.log("val_loss", loss)
        
        self.get_metrics(y_hat, y, 'val')

        return loss

    def test_step(self, batch, batch_idx):
        x, y, *_ = batch
        y_hat = self.forward(x)
        loss = self.get_loss(y_hat, y)
        self.log("test_loss", loss)
        
        self.get_metrics(y_hat, y, 'test')
        
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == '__main__':
    TRAIN_BATCH_SIZE = 1024
    TEST_BATCH_SIZE = 1024
    EPOCHS = 50

    dataset = MoNADataset(fingerprint_type='ECFP')
    train_set, test_set, extra = utils.data.random_split(dataset, [0.8, 0.2, 0], generator=Generator().manual_seed(2022))

    num_workers = multiprocessing.cpu_count()
    train_loader = utils.data.DataLoader(train_set, num_workers=num_workers, batch_size=TRAIN_BATCH_SIZE) 
    test_loader = utils.data.DataLoader(test_set, num_workers=num_workers, batch_size=TEST_BATCH_SIZE) 

    model = FFN(lr=1e-3, hidden_layer_sizes=[4096], input_size=len(train_set[0][0]), output_size=1000)

    trainer = pl.Trainer(max_epochs=EPOCHS, auto_select_gpus = True, auto_scale_batch_size=True)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)