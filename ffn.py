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
            hidden_layers.append(nn.Dropout(0.5))
            hidden_layers.append(nn.ReLU())
        else:
            hidden_layers.append(nn.Linear(self.INPUT_SIZE, self.OUTPUT_SIZE))

        # Weird syntax
        for idx, _size in enumerate(hidden_layer_sizes[1:], start=1):
            hidden_layers.append(nn.Linear(hidden_layer_sizes[idx-1], hidden_layer_sizes[idx]))
            hidden_layers.append(nn.BatchNorm1d(hidden_layer_sizes[idx]))
            hidden_layers.append(nn.Dropout(0.5))
            hidden_layers.append(nn.ReLU())
            
        hidden_layers.append(nn.Linear(hidden_layer_sizes[-1], self.OUTPUT_SIZE))

        # Final activation
        # hidden_layers.append(nn.Softmax(dim=1))
        
        
        self.ffn = nn.Sequential(*hidden_layers)


    def forward(self, x):
        return self.ffn(x)

    def get_loss(self, y_hat, y, mask=None):
        #loss = 1 - F.cosine_similarity(y_hat, y, axis=1).mean()
        #loss = F.cross_entropy(y_hat, np.argmax(y, axis=1)).mean()
        loss = (F.mse_loss(y_hat, y, reduction='none')*mask).sum()
        return loss
    
    def get_metrics(self, y_hat, y, log_name):
        # Cosine similarity between (un)normalized peaks and model output 
        self.log(f"cosineSim/{log_name}", F.cosine_similarity(y_hat, y).mean(), prog_bar=True)

        # Mean AUROC of top-1 peak vs all other peaks across molecules
        self.log(f"mAUROC/{log_name}", np.mean([sklearn.metrics.roc_auc_score(F.one_hot(np.argmax(y[i]), self.OUTPUT_SIZE).reshape(-1,1), y_hat[i]) for i in range(len(y))]), prog_bar=True)
        # Mean top-1 peak Rank across molecules
        self.log(f"peakRank/{log_name}", np.mean([float(i) for i in np.argmax(y_hat, axis=1)]), prog_bar=True)

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        y_hat = self.forward(x)
        loss = self.get_loss(y_hat, y, mask=mask)
        
        self.log("loss/train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        y_hat = self.forward(x)
        loss = self.get_loss(y_hat, y, mask=mask)
        self.log("loss/val", loss)
        
        self.get_metrics(y_hat, y, 'val')

        return loss

    def test_step(self, batch, batch_idx):
        x, y, *_ = batch
        y_hat = self.forward(x)
        loss = self.get_loss(y_hat, y)
        self.log("loss/test", loss)
        
        self.get_metrics(y_hat, y, 'test')
        
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == '__main__':
    TRAIN_BATCH_SIZE = 1024
    TEST_BATCH_SIZE = 1024
    EPOCHS = 50

    dataset = MoNADataset(fingerprint_type='MACCS', bayesian_mask=True, sparse_weight=0.1)
    train_set, test_set, extra = utils.data.random_split(dataset, [0.8, 0.2, 0], generator=Generator().manual_seed(2022))
    
    print(f"INPUT_SIZE: {len(train_set[0][0])}")

    num_workers = multiprocessing.cpu_count()
    train_loader = utils.data.DataLoader(train_set, num_workers=num_workers, batch_size=TRAIN_BATCH_SIZE) 
    test_loader = utils.data.DataLoader(test_set, num_workers=num_workers, batch_size=TEST_BATCH_SIZE) 

    model = FFN(lr=1e-3, hidden_layer_sizes=[4096], input_size=len(train_set[0][0]), output_size=1000)
    
    METRIC = 'peakRank/val'
    MODE = 'min'
    TRIAL_DIR = 'logs/ffn2'
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        TRIAL_DIR, monitor=METRIC, mode=MODE, filename='best'
    )
    callbacks=[checkpoint_callback, pl.callbacks.ModelSummary(max_depth=-1), 
               pl.callbacks.EarlyStopping(monitor=METRIC, mode=MODE, patience=5, min_delta=0.001)]
    logger = pl.loggers.TensorBoardLogger(TRIAL_DIR)
    
    # Remove previous Tensorboard statistics
    if os.path.exists(TRIAL_DIR):
        os.system(f"rm -r {TRIAL_DIR}")
    os.system(f'mk -p {TRIAL_DIR}')

    trainer = pl.Trainer(max_epochs=EPOCHS, auto_select_gpus = True, auto_scale_batch_size=True, 
                         callbacks=callbacks, logger=logger)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)