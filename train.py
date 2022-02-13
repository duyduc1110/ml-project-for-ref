import pandas as pd, numpy as np
import h5py, logging, argparse
import torch, torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime


class BruceCNNModel(pl.LightningModule):
    def __init__(self, param1=1, param2=2):
        super().__init__()
        self.save_hyperparameters()
        self.save_hyperparameters({'duc': 1, 'bon':2})
        
        # CNN Layer
        self.cnn1 = nn.Conv1d(1, 8, 8, padding='same')
        self.ln1 = nn.LayerNorm(524)
        self.avgpool1 = nn.AvgPool1d(8)
        self.cnn2 = nn.Conv1d(1, 8, 16, padding='same')
        self.ln2 = nn.LayerNorm(524)
        self.avgpool2= nn.AvgPool1d(8)
        self.flatten = nn.Flatten()
        self.cnn_out = nn.Linear(1040, 64)
        self.dropout_cnn = nn.Dropout(0.1)

        # FCN Layer
        self.fc1 = nn.Linear(64, 512)
        self.fc1_act = nn.GELU()
        self.fc2 = nn.Linear(512, 64)
        self.fc2_act = nn.GELU()
        self.layernorm_fcn = nn.LayerNorm(64)
        self.drop_fcn = nn.Dropout(0.1)

        # Cls output
        self.cls_out = nn.Linear(64, 1)

        # Regression output
        self.cls_embed = nn.Embedding(2, 64, padding_idx=0)
        self.layernorm_rgs = nn.LayerNorm(64)
        self.rgs_out = nn.Linear(64, 1)

        # Loss function
        self.cls_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.1275]))
        self.rgs_loss_fn = nn.L1Loss()

        # Loss weights
        self.loss_weights = torch.tensor(LOSS_WEIGHTS).float()

        # Metrics to log
        self.acc = torchmetrics.Accuracy()
        self.auc = torchmetrics.AUC()

    def forward(self, inputs):
        b, f = inputs.shape
        # CNN forward
        x1 = self.avgpool1(self.ln1(self.cnn1(inputs.reshape(b, 1, f).float())))
        x2 = self.avgpool2(self.ln2(self.cnn2(inputs.reshape(b, 1, f).float())))
        x = torch.cat([x1, x2], -1)
        x = self.flatten(x)
        x = F.gelu(self.cnn_out(x))
        x = self.dropout_cnn(x)

        # FCN
        t = self.fc1_act(self.fc1(x))
        t = self.fc2_act(self.fc2(t))
        x = self.layernorm_fcn(x + t)
        x = self.drop_fcn(x)

        # Classification output
        cls_out = self.cls_out(x)

        # Regression output
        bi_cls = (cls_out > 0.5).long()
        cls_embed = self.cls_embed(bi_cls)
        rgs_out = self.layernorm_rgs(cls_out + cls_embed)
        rgs_out = self.rgs_out(rgs_out)

        return cls_out, rgs_out

    def loss(self, inputs, cls_true, rgs_true):
        cls_out, rgs_out = inputs
        return self.cls_loss_fn(cls_out, cls_true.float()), self.rgs_loss_fn(rgs_out, rgs_true)

    def training_step(self, batch, batch_idx):
        inputs, cls_labels, rgs_labels = batch
        outs = self(inputs)
        cls_loss, rgs_loss = self.loss(outs, cls_labels, rgs_labels)

        loss = cls_loss * self.loss_weights[0] + rgs_loss * self.loss_weights[1]
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)


class BruceDataset(Dataset):
    def __init__(self, inputs, cls_labels=None, rgs_labels=None):
        super().__init__()
        self.inputs = inputs
        self.cls_labels = cls_labels
        self.rgs_labels = rgs_labels

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        if self.cls_labels is None:
            return torch.tensor(self.inputs[idx])
        else:
            return torch.tensor(self.inputs[idx]), \
                   torch.tensor(self.cls_labels[idx]), \
                   torch.tensor(self.rgs_labels[idx])


def read_data(path):
    f = h5py.File(path)
    return f


def preprocessing_data(arr, normalize=True):
    if normalize:
        # Data normalization
        return (arr - arr.min()) / (arr.max() - arr.min())
    else:
        # Data standardlization
        return (arr - arr.mean()) / arr.std()


def split_data(x, y1, y2):
    return train_test_split(x, y1, y2, test_size=0.2, stratify=y1)


def get_data(path, sample=False):
    f = read_data(path)
    if sample:
        data = f['Samples_big'][:100]
    else:
        data = f['Samples_big'][:]

    train_rgs_labels = data[:, 526].reshape(-1, 1)
    train_cls_labels = (train_rgs_labels > 0).reshape(-1, 1)
    train_inputs = preprocessing_data(data[:, :524], True)

    return split_data(train_inputs, train_cls_labels, train_rgs_labels)


def get_args():
    model_parser = argparse.ArgumentParser()

    # Model argumentss
    model_parser.add_argument('--logging_level', default='INFO', type=str, help="Set logging level")
    model_parser.add_argument('--model_path', default=None, type=str, help="Model path")
    model_parser.add_argument('--run_name', default=None, type=str, help="Run name to put in WanDB")
    model_parser.add_argument('--data_path', default='data.mat', type=str, help='Data path')
    model_parser.add_argument('--sample', default=True, type=bool, help='Sample to test data and model')

    # Trainer arguments
    model_parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    model_parser.add_argument('--training_step', default=100000, type=int, help='Training steps')
    model_parser.add_argument('--batch_size', default=64, type=int, help='Batch size per device')
    model_parser.add_argument('--log_step', default=100, type=int, help='Steps per log')
    model_parser.add_argument('--gpu', default=0, type=int, help='Use GPUs')
    model_parser.add_argument('--epoch', default=10, type=int, help='Number of epoch')

    args = model_parser.parse_args()
    return args


INIT_LR = 1e-3
LOSS_WEIGHTS = [1, 1]
EPOCH = 100
SCHEDULER_EPOCH = 20
SCHEDULER_RATE = 0.9
CLASS_W = [{0: 0.85, 1: 0.15}, None]
DATETIME_NOW = datetime.now().strftime('%Y-%m-%d_%H-%M')


if __name__ == '__main__':
    # Get arguments & logger
    args = get_args()
    logging.basicConfig(level=args.logging_level)
    logger = logging.getLogger('model')

    # Generate model
    model = BruceCNNModel()

    # Get data
    train_x, val_x, train_y_cls, val_y_cls, train_y_rgs, val_y_rgs = get_data(args.data_path, args.sample)

    # Create Dataloader
    train_dataset = TensorDataset(torch.tensor(train_x),
                                   torch.tensor(train_y_cls),
                                   torch.tensor(train_y_rgs))
    val_dataset = TensorDataset(torch.tensor(val_x),
                                  torch.tensor(val_y_cls),
                                  torch.tensor(val_y_rgs))
    #train_dataset = BruceDataset(inputs=train_x, cls_labels=train_y_cls, rgs_labels=train_y_rgs)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    # Init Trainer
    wandb_logger = WandbLogger(project='Rocsole_DILI', name=f'CNN-{DATETIME_NOW}')
    wandb_logger.watch(model, log='all')

    trainer = pl.Trainer(
        logger=wandb_logger,
        gpus=args.gpu,
        log_every_n_steps=args.log_step,
        max_epochs=args.epoch,
    )

    trainer.fit(model, train_dataloader, val_dataloader)

    pass