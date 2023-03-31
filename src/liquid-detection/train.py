import h5py, logging, argparse, getpass, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import torch, torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb

from torch.utils.data import DataLoader, Dataset, TensorDataset
# from model import BruceModel
from preprocess_data import read_teflon
from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from datetime import datetime
from weakref import ReferenceType


def normalize_data(data):
    return np.log(np.sqrt(data))


def preprocess_input(data):
    t1, t2, t3 = data.reshape(-1, 32, 3, 8).transpose(2, 0, 1, 3)
    t2 = np.roll(t2, 16, axis=1)  # swap bot and top
    new = np.concatenate([
        np.pad(t1, ((0, 0), (0, 32), (0, 0))),
        np.pad(t2, ((0, 0), (16, 16), (0, 0))),
        np.pad(t3, ((0, 0), (32, 0), (0, 0))),
    ], axis=-1)
    return new

process_label = lambda idx: [1] * idx + [0] * (64 - idx)

if __name__ == '__main__':
    df = read_teflon()
    df.to_csv('teflon_data.csv', sep='\t', index=False)

    df = pd.read_csv('teflon_data.csv', sep='\t').drop(index=483).reset_index(drop=True)  # There is data error in measurement 483
    data_4e = df.iloc[:, :1792].values.reshape(-1, 32, 24 + 32)[:, :, :24]
    data_1e_middle = df.iloc[:, :1792].values.reshape(-1, 32, 24 + 32)[:, :, 24:]

    # data_4e = normalize_data(data_4e)
    inputs = preprocess_input(data_4e)

    labels = np.array(df.y.apply(process_label).values.tolist())

    print('Hi')
