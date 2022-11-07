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
        OUTPUT_SIZE = 1000

        self.ffn = nn.Sequential(
            nn.Linear(INPUT_SIZE, 4096), nn.ReLU(), 
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Linear(4096, OUTPUT_SIZE), nn.ReLU()

                                    )
    def forward(self, x):
        return self.ffn(x)

    def get_loss(self, y_hat, y, mask=None):
        loss = F.mse_loss(y_hat, y, reduction='none')
        if mask is not None:
            loss *= mask;
            return loss.sum() / mask.sum();
        return loss.mean();

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        y_hat = self.forward(x)
        loss = self.get_loss(y_hat, y)
        
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        y_hat = self.forward(x)
        loss = self.get_loss(y_hat, y)
        
        self.log("test_loss", loss)
        self.log("test_cosine_sim", F.cosine_similarity(y_hat, y).mean(), prog_bar=True)

        return loss



    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


TRAIN_BATCH_SIZE = 8192

model = Vanilla_FFN()

dataset = MoNADataset()
train_set, test_set, extra = utils.data.random_split(dataset, [0.2, 0.05, 0.75], generator=Generator().manual_seed(2022))

train_loader = utils.data.DataLoader(train_set, num_workers=12, batch_size=TRAIN_BATCH_SIZE) 
test_loader = utils.data.DataLoader(test_set, batch_size=len(test_set)) 

trainer = pl.Trainer(max_epochs=5, auto_select_gpus = True, auto_scale_batch_size=True)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)

