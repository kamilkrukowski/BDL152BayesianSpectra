import os


from torch import optim, nn, utils, Tensor, Generator
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import pytorch_lightning as pl


from dataloader import MoNADataset


# define the LightningModule
class Vanilla_FFN(pl.LightningModule):
    def __init__(self):
        super().__init__()

        INPUT_SIZE = 167
        OUTPUT_SIZE = 1,000

        self.ffn = nn.Sequential(
            nn.Linear(INPUT_SIZE, 16384), nn.ReLU(), 
            nn.Linear(16384, 8096), nn.ReLU(),
            nn.Linear(8096, 2048), nn.ReLU(),
            nn.Linear(2048, 1000), nn.ReLU()

                                    )
        
        self.loss = F.mse_loss

    def forward(self, x):
        return self.ffn(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        
        self.log("test_loss", loss)
        self.log("test_cosine_sim", F.cosine_similarity(y_hat, y).mean(), prog_bar=True)

        return loss



    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


TRAIN_BATCH_SIZE = 1024

model = Vanilla_FFN()

dataset = MoNADataset()
train_set, test_set, extra = utils.data.random_split(dataset, [0.8, 0.2, 0.00], generator=Generator().manual_seed(2022))

train_loader = utils.data.DataLoader(train_set, num_workers=12, batch_size=1024) 
test_loader = utils.data.DataLoader(test_set, batch_size=len(test_set)) 

trainer = pl.Trainer(max_epochs=5, auto_select_gpus = True, auto_scale_batch_size=True)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)

