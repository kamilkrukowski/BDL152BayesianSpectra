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

        self.INPUT_SIZE = 167
        self.PADDED_SIZE = 256
        self.OUTPUT_SIZE = 1000

        self.embed_attn = nn.MultiheadAttention(self.PADDED_SIZE, 128, 0.1)

        hidden_layer_sizes = [4096]
        hidden_layers = []
    
        if len(hidden_layer_sizes) > 0:
            hidden_layers.append(nn.Linear(self.PADDED_SIZE, hidden_layer_sizes[0]))
            hidden_layers.append(nn.BatchNorm1d(hidden_layer_sizes[0]))
            hidden_layers.append(nn.Dropout(0.1))
        else:
            hidden_layers.append(nn.Linear(self.INPUT_SIZE, self.OUTPUT_SIZE))


        for idx, _size in enumerate(hidden_layer_sizes[1:], start=1):
            hidden_layers.append(nn.Linear(hidden_layer_sizes[idx-1], hidden_layer_sizes[idx]))
            hidden_layers.append(nn.BatchNorm1d(hidden_layer_sizes[idx]))
            hidden_layers.append(nn.Dropout(0.5))
            
        hidden_layers.append(nn.Linear(hidden_layer_sizes[-1], self.OUTPUT_SIZE))
        
        self.ffn = nn.Sequential(*hidden_layers)


    def forward(self, x):
        x = F.pad(x, (0, self.PADDED_SIZE - self.INPUT_SIZE), 'constant', 0)
        x, _ = self.embed_attn(x, x, x)
        return self.ffn(x)

    def get_loss(self, y_hat, y, mask=None):
        loss = 1 - F.cosine_similarity(y_hat, y, axis=1)
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
        self.log("test_cosine_sim", F.cosine_similarity(y_hat, y).mean(), prog_bar=True, on_epoch=True)

        return loss



    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-2, weight_decay=1e-6)
        return optimizer


TRAIN_BATCH_SIZE = 1024
TEST_BATCH_SIZE = 1024

model = Vanilla_FFN()

dataset = MoNADataset()
train_set, test_set, extra = utils.data.random_split(dataset, [0.8, 0.2, 0.0], generator=Generator().manual_seed(2022))

train_loader = utils.data.DataLoader(train_set, num_workers=12, batch_size=TRAIN_BATCH_SIZE) 
test_loader = utils.data.DataLoader(test_set, batch_size=TEST_BATCH_SIZE) 

trainer = pl.Trainer(max_epochs=50, auto_select_gpus = True, auto_scale_batch_size=True)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)

