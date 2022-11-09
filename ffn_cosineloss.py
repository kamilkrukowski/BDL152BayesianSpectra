import os


from torch import optim, nn, utils, Tensor, Generator
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import pytorch_lightning as pl


from dataloader import MoNADataset


# define the LightningModule
class Vanilla_FFN(pl.LightningModule):
    def __init__(self, lr, n_hidden_layers=None, hidden_layer_width=None, hidden_layer_sizes=None, weight_decay=None):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay

        self.INPUT_SIZE = 167
        self.PADDED_SIZE = 256
        self.OUTPUT_SIZE = 1000

        assert ((n_hidden_layers is not None) and (hidden_layer_width is not None)) or (hidden_layer_sizes is not None), 'Specify Hidden Layers';

        if hidden_layer_sizes is None:
            hidden_layer_sizes = [hidden_layer_width]*n_hidden_layers

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
        x = F.pad(x, (0, self.PADDED_SIZE-self.INPUT_SIZE), 'constant', 0)
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
        
        self.log("val_loss", loss)
        self.log("val_cosine_sim", F.cosine_similarity(y_hat, y).mean(), prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, mask = batch
        y_hat = self.forward(x)
        loss = self.get_loss(y_hat, y)
        
        self.log("test_loss", loss)
        self.log("test_cosine_sim", F.cosine_similarity(y_hat, y).mean(), prog_bar=True, on_epoch=True)
        return loss


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer


if __name__ == '__main__':
    TRAIN_BATCH_SIZE = 1024
    TEST_BATCH_SIZE = 1024

    model = Vanilla_FFN(lr=1e-3, hidden_layer_sizes=[4096])

    dataset = MoNADataset()
    train_set, test_set, extra = utils.data.random_split(dataset, [0.8, 0.2, 0.0], generator=Generator().manual_seed(2022))

    train_loader = utils.data.DataLoader(train_set, num_workers=12, batch_size=TRAIN_BATCH_SIZE) 
    test_loader = utils.data.DataLoader(test_set, batch_size=TEST_BATCH_SIZE) 

    trainer = pl.Trainer(max_epochs=50, auto_select_gpus = True, auto_scale_batch_size=True)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)

