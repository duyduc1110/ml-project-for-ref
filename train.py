import h5py, logging, argparse, getpass, pandas as pd, numpy as np
import torch, torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset, TensorDataset
from model import BruceCNNModel, BruceRNNModel
from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from datetime import datetime


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


def get_data(path, no_sample):
    f = read_data(path)
    if no_sample:
        data = f['Samples_big'][:]
    else:
        data = f['Samples_big'][:2000]

    train_rgs_labels = data[:, 526].reshape(-1, 1) * 10
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
    model_parser.add_argument('--no_sample', action='store_true', help='Sample to test data and model')
    model_parser.add_argument('--bi_di', action='store_true', help='Bi-directional for RNN')
    model_parser.add_argument('--hidden_size', default=256, type=int, help='Hidden size')
    model_parser.add_argument('--num_lstm_layer', default=1, type=int, help='Number of LSTM layer')

    # Trainer arguments
    model_parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    model_parser.add_argument('--training_step', default=100000, type=int, help='Training steps')
    model_parser.add_argument('--batch_size', default=128, type=int, help='Batch size per device')
    model_parser.add_argument('--log_step', default=100, type=int, help='Steps per log')
    model_parser.add_argument('--gpu', default=0, type=int, help='Use GPUs')
    model_parser.add_argument('--epoch', default=5, type=int, help='Number of epoch')

    args = model_parser.parse_args()
    return args


def get_predict(model, dataloaders):
    y_trues = []
    predicts = []

    for dataloader in dataloaders:
        for batch in dataloader:
            inputs, cls_labels, rgs_labels = batch
            cls_out, rgs_out = model(inputs)

            y_trues.extend(rgs_labels.reshape(-1).tolist())
            predicts.extend(rgs_out.reshape(-1).tolist())

    return y_trues, predicts


INIT_LR = 1e-3
LOSS_WEIGHTS = [0.7, 0.3]
EPOCH = 100
SCHEDULER_EPOCH = 20
SCHEDULER_RATE = 0.9
CLASS_W = [{0: 0.85, 1: 0.15}, None]
DATETIME_NOW = datetime.now().strftime('%Y%m%d_%H%M')


if __name__ == '__main__':
    # Get arguments & logger
    args = get_args()
    args.loss_weights = LOSS_WEIGHTS # set loss weights
    logging.basicConfig(level=args.logging_level)
    logger = logging.getLogger('model')

    # Generate model
    model = BruceCNNModel(args)

    # Get data
    train_x, val_x, train_y_cls, val_y_cls, train_y_rgs, val_y_rgs = get_data(args.data_path, args.no_sample)

    # Create Dataloader
    train_dataset = TensorDataset(torch.FloatTensor(train_x),
                                  torch.FloatTensor(train_y_cls),
                                  torch.FloatTensor(train_y_rgs))
    val_dataset = TensorDataset(torch.FloatTensor(val_x),
                                torch.FloatTensor(val_y_cls),
                                torch.FloatTensor(val_y_rgs))
    #train_dataset = BruceDataset(inputs=train_x, cls_labels=train_y_cls, rgs_labels=train_y_rgs)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Init Logger
    MODEL_NAME = f'CNN-{DATETIME_NOW}_{getpass.getuser()}'
    wandb_logger = WandbLogger(project='Rocsole_DILI', name=MODEL_NAME)
    wandb_logger.watch(model, log='all')

    # Init Callbacks
    profiler = AdvancedProfiler()
    model_checker = ModelCheckpoint(monitor='train/rgs_loss',
                                    mode='min',
                                    dirpath='./model_checkpoint/',
                                    filename=MODEL_NAME,
                                    verbose=True)

    # Init Pytorch Lightning Profiler
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[model_checker],
        #profiler=profiler,
        gpus=args.gpu,
        log_every_n_steps=args.log_step,
        max_epochs=args.epoch,
        deterministic=True,
    )

    trainer.fit(model, train_dataloader, val_dataloader)
    model_checker.best_model_path

    y_trues, predicts = get_predict(model, (train_dataloader, val_dataloader))
    df = pd.DataFrame(np.array([y_trues, predicts]).T, columns=['True', 'Predicted'])
    df.to_csv(f'./predicts/{MODEL_NAME}.csv', index=False)


