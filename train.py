import pandas as pd, numpy as np, tensorflow as tf, tensorflow.keras as K
import h5py, logging, argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl



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

    def forward(self, inputs):
        b, f = inputs.shape
        # CNN forward
        x1 = self.avgpool1(self.ln1(self.cnn1(inputs.reshape(b, 1, f))))
        x2 = self.avgpool2(self.ln2(self.cnn2(inputs.reshape(b, 1, f))))
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

    def loss(self, inputs, labels):
        cls_out, rgs_out = inputs
        cls_true, rgs_true = labels
        return self.cls_loss_fn(cls_out, cls_true), self.rgs_loss_fn(rgs_out, rgs_true)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outs = self(inputs)
        cls_loss, rgs_loss = self.loss(outs, labels)

        loss = cls_loss * self.loss_weights[0] + rgs_loss * self.loss_weights[1]
        return loss


def read_data(path='/drive/MyDrive/samples_approxerror_equinor_PIG_010720.mat'):
    f = h5py.File(path)
    return f


def preprocessing_data(arr, normalize=True):
    if normalize:
        # Data normalization
        return (arr - arr.min()) / (arr.max() - arr.min())
    else:
        # Data standardlization
        return (arr - arr.mean()) / arr.std()


def get_data(path):
    f = read_data()
    data = f['Samples_big'][:]

    train_rgs_labels = data[:, 526].reshape(-1, 1)
    train_cls_labels = (train_rgs_labels > 0).reshape(-1, 1)
    train_inputs = preprocessing_data(data[:, :524], True)

    return train_inputs, train_cls_labels, train_rgs_labels


def get_args():
    model_parser = argparse.ArgumentParser()

    # Model argumentss
    model_parser.add_argument('--logging_level', default='INFO', type=str, help="Set logging level")
    model_parser.add_argument('--model_path', default=None, type=str, help="Model path")
    model_parser.add_argument('--run_name', default=None, type=str, help="Run name to put in WanDB")
    model_parser.add_argument('--data_path', default='/drive/MyDrive/', type=str, help='Data path')

    # Trainer arguments
    model_parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    model_parser.add_argument('--training_step', default=100000, type=int, help='Training steps')
    model_parser.add_argument('--batch_size', default=64, type=int, help='Batch size per device')
    model_parser.add_argument('--log_step', default=100, type=int, help='Steps per log')

    args = model_parser.parse_args()
    return args


INIT_LR = 1e-3
LOSS_WEIGHTS = [1, 1]
BATCH_SIZE = 2048
EPOCH = 100
SCHEDULER_EPOCH = 20
SCHEDULER_RATE = 0.9
CLASS_W = [{0: 0.85, 1: 0.15}, None]
METRICS = ['accuracy', K.metrics.AUC()]


if __name__ == '__main__':
    # Get arguments & logger
    args = get_args()
    logging.basicConfig(level=args.logging_level)
    logger = logging.getLogger('model')

    # Generate model
    model = BruceCNNModel()

    # Get data
    train_inputs, train_cls_labels, train_rgs_labels = get_data(args.data_path)

    pass