import os


from torch import optim, nn, utils, Tensor, Generator, cuda
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import pytorch_lightning as pl
import optuna
from optuna.pruners import HyperbandPruner


from optuna_callback import PyTorchLightningPruningCallback
from dataloader import MoNADataset
from ffn_cosineloss import Vanilla_FFN


MODEL_DIR = os.path.join('','trials')
TEST_BATCH_SIZE = 1024
EPOCHS = 25


def objective(trial):
    TRIAL_DIR = os.path.join(MODEL_DIR, f"trial_{trial.number}")
    # PyTorch Lightning will try to restore model parameters from previous trials if checkpoint
    # filenames match. Therefore, the filenames for each trial must be made unique.
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        TRIAL_DIR, monitor="val_cosine_sim", mode='max'
    )

    # The default logger in PyTorch Lightning writes to event files to be consumed by
    # TensorBoard. We create a simple logger instead that holds the log in memory so that the
    # final accuracy can be obtained after optimization. When using the default logger, the
    # final accuracy could be stored in an attribute of the `Trainer` instead.
    logger = pl.loggers.TensorBoardLogger(TRIAL_DIR)

    TRAIN_BATCH_SIZE = trial.suggest_int('train_batch_size', 32, 4096, log=True)

    train_loader = utils.data.DataLoader(TRAIN_SET, num_workers=12, batch_size=TRAIN_BATCH_SIZE) 

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, PyTorchLightningPruningCallback(trial, monitor="val_cosine_sim"),
            pl.callbacks.ModelSummary(max_depth=-1), pl.callbacks.EarlyStopping(monitor="val_cosine_sim", mode="max", patience=5, min_delta=0.001)],
        gpus=0 if cuda.is_available() else None,
        max_epochs=EPOCHS
    )

    model = Vanilla_FFN(lr=trial.suggest_float('lr', 1.0e-5, 1.0e-1, log=True), n_hidden_layers=trial.suggest_int('n_hidden_units', 1, 1), 
            hidden_layer_width=trial.suggest_int('hidden_layer_width', 256, 4096, log=True), weight_decay=trial.suggest_float('weight_decay', 1.0e-10, 1.0e-4, log=True))
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return trainer.test(dataloaders=test_loader, ckpt_path='best')[-1]['test_cosine_sim']

dataset = MoNADataset()
TRAIN_SET, val_set, test_set = utils.data.random_split(dataset, [0.8, 0.1, 0.1], generator=Generator().manual_seed(2022))

val_loader = utils.data.DataLoader(val_set, num_workers=12, batch_size=TEST_BATCH_SIZE) 
test_loader = utils.data.DataLoader(test_set, num_workers=12, batch_size=TEST_BATCH_SIZE) 

study = optuna.create_study(direction="maximize", pruner=HyperbandPruner())
study.optimize(objective, n_trials=100, timeout=None)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
