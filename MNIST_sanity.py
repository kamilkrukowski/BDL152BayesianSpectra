

import torch
from torch import Generator, utils
from torchvision.datasets import MNIST
from torchvision import transforms as T
import pytorch_lightning as pl


from mean_field_vi import BayesianNetwork



TRAIN_BATCH_SIZE = 1024
TEST_BATCH_SIZE = 1024

model = BayesianNetwork(lr=1e-3, hidden_layer_sizes=[28*28, 64],
            INPUT_SIZE=28*28, OUTPUT_SIZE=10)

train_set = MNIST('data', download=True, train=True, transform=T.Compose([T.ToTensor(), T.Lambda(torch.flatten)]))
test_set = MNIST('data', download=True, train=False, transform=T.Compose([T.ToTensor(), T.Lambda(torch.flatten)]))

a = next(iter(train_set))[0]
print(a.shape)


train_loader = utils.data.DataLoader(train_set, num_workers=12, batch_size=TRAIN_BATCH_SIZE) 
test_loader = utils.data.DataLoader(test_set, num_workers=12, batch_size=TEST_BATCH_SIZE) 

trainer = pl.Trainer(max_epochs=50, auto_select_gpus = True, auto_scale_batch_size=True)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)
