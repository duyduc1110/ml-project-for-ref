import h5py, logging, argparse, getpass, pandas as pd, numpy as np
import torch, torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset, TensorDataset
from model import BruceModel
from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from datetime import datetime


class ParseAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        print('%r %r %r' % (namespace, values, option_string))
        values = list(map(int, values.split()))
        setattr(namespace, self.dest, values)


class BruceDataset(Dataset):
    def __init__(self, inputs, cls_labels=None, rgs_labels=None, seq_len=32):
        super().__init__()
        self.inputs = inputs
        self.cls_labels = cls_labels
        self.rgs_labels = rgs_labels
        self.seq_len = seq_len

    def __len__(self):
        return self.inputs.shape[0] - self.seq_len

    def __getitem__(self, idx):
        if self.cls_labels is None:
            return torch.tensor(self.inputs[idx : idx+self.seq_len])
        else:
            return torch.tensor(self.inputs[idx : idx+self.seq_len]), \
                   torch.tensor(self.cls_labels[idx : idx+self.seq_len]), \
                   torch.tensor(self.rgs_labels[idx : idx+self.seq_len])


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


def get_data(path, no_sample, normalize=True):
    f = read_data(path)
    if no_sample:
        data = f['Samples_big'][:]
    else:
        data = f['Samples_big'][:2000]

    train_rgs_labels = data[:, 526].reshape(-1, 1) * 10
    train_cls_labels = (train_rgs_labels > 0).reshape(-1, 1)
    train_inputs = preprocessing_data(data[:, :524], normalize)

    return split_data(train_inputs, train_cls_labels, train_rgs_labels)


def get_args():
    model_parser = argparse.ArgumentParser()

    # Model argumentss
    model_parser.add_argument('--logging_level', default='INFO', type=str, help="Set logging level")
    model_parser.add_argument('-bb', '--backbone', default='cnn', type=str, help='Model backbone: cnn, lstm, att')
    model_parser.add_argument('--data_path', default='data.mat', type=str, help='Data path')
    model_parser.add_argument('--rgs_loss', default='mae', type=str, help="Regression loss, default is MAE")
    model_parser.add_argument('--normalize', default=0, type=int, help='Normalize data if used, otherwise standardize ')
    model_parser.add_argument('--no_sample', action='store_true', help='Sample to test data and model')
    model_parser.add_argument('--hidden_size', default=256, type=int, help='Hidden size')
    model_parser.add_argument('-sl', '--seq_len', default=32, type=int, help='Sequence len')
    model_parser.add_argument('--cls_w', default=0.8, type=float, help='Classification weight')
    model_parser.add_argument('-co', '--core_out', default=256, type=int, help='Core output channel')
    model_parser.add_argument('--initializer_range', default=0.02, type=float, help='Initializer range')

    # CNN args
    model_parser.add_argument('-nc', '--num_cnn', default=1, type=int, help='Number of CNN Layer')
    model_parser.add_argument('-ks', '--kernel_size', default=[3], action=ParseAction, help='Kernel size for each CNN layer')
    model_parser.add_argument('-oc', '--output_channel', default=[8], action=ParseAction, help='Output channel for each CNN layer')

    # LSTM args
    model_parser.add_argument('--bi_di', action='store_true', help='Bi-directional for RNN')
    model_parser.add_argument('--num_lstm_layer', default=1, type=int, help='Number of LSTM layer')

    # Trainer arguments
    model_parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    model_parser.add_argument('--find_lr', action='store_true', help='Find best learning rate')
    model_parser.add_argument('--batch_size', default=128, type=int, help='Batch size per device')
    model_parser.add_argument('--log_step', default=100, type=int, help='Steps per log')
    model_parser.add_argument('--gpu', default=0, type=int, help='Use GPUs')
    model_parser.add_argument('--num_epoch', default=10, type=int, help='Number of epoch')

    args = model_parser.parse_args()
    return args


def get_predict(model, dataloaders):
    y_trues = []
    rgs = []
    cls = []

    for dataloader in dataloaders:
        for batch in dataloader:
            inputs, cls_labels, rgs_labels = batch
            cls_out, rgs_out = model(inputs)

            y_trues.extend(rgs_labels.reshape(-1).tolist())
            cls.extend(torch.sigmoid(cls_out.reshape(-1)).tolist())
            rgs.extend(rgs_out.reshape(-1).tolist())

    final_predicts = (np.array(cls) >= 0.5) * np.array(rgs)

    return y_trues, cls, rgs, final_predicts



INIT_LR = 1e-3
EPOCH = 100
SCHEDULER_EPOCH = 20
SCHEDULER_RATE = 0.9
CLASS_W = [{0: 0.85, 1: 0.15}, None]
DATETIME_NOW = datetime.now().strftime('%Y%m%d_%H%M')


if __name__ == '__main__':
    # Get arguments & logger
    args = get_args()
    args.loss_weights = [args.cls_w, 1 - args.cls_w] # set loss weights
    logging.basicConfig(level=args.logging_level)
    logger = logging.getLogger('model')
    logger.info(args.__dict__)

    # Generate model
    model = BruceModel(**args.__dict__)
    logger.info(model)

    # Get data
    train_x, val_x, train_y_cls, val_y_cls, train_y_rgs, val_y_rgs = get_data(args.data_path,
                                                                              args.no_sample,
                                                                              args.normalize)

    # Create Dataloader

    train_dataset = TensorDataset(torch.FloatTensor(train_x),
                                  torch.FloatTensor(train_y_cls),
                                  torch.FloatTensor(train_y_rgs))
    val_dataset = TensorDataset(torch.FloatTensor(val_x),
                                torch.FloatTensor(val_y_cls),
                                torch.FloatTensor(val_y_rgs))
    '''
    train_dataset = BruceDataset(inputs=train_x, cls_labels=train_y_cls, rgs_labels=train_y_rgs)
    '''

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Init Logger
    MODEL_NAME = f'CNN-{DATETIME_NOW}_{getpass.getuser()}'
    wandb_logger = WandbLogger(project='Rocsole_DILI', name=MODEL_NAME)
    wandb_logger.watch(model, log='all')

    # Init Callbacks
    profiler = AdvancedProfiler()
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    early_stop_callback = EarlyStopping(monitor='val/rgs_loss',
                                        mode='min',
                                        patience=8,
                                        verbose=True)
    model_checker = ModelCheckpoint(monitor='val/rgs_loss',
                                    mode='min',
                                    dirpath='./model_checkpoint/',
                                    filename=MODEL_NAME + '_{epoch:02d}-{val_rgs_loss:.2f}',
                                    verbose=True)

    # Init Pytorch Lightning Profiler
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[lr_monitor, early_stop_callback, model_checker],
        #profiler=profiler,
        gpus=args.gpu,
        log_every_n_steps=args.log_step,
        max_epochs=args.num_epoch,
        deterministic=True,
    )

    if args.find_lr:
        lr_finder = trainer.tuner.lr_find(model, train_dataloader, val_dataloader,  min_lr=1e-5, max_lr=0.1)
        print(lr_finder.results)

        fig = lr_finder.plot(suggest=True, show=True)

        new_lr = lr_finder.suggestion()
        print('Suggested learning rate: ', new_lr)
        model.hparams.lr = new_lr

    # Fit training data
    trainer.fit(model, train_dataloader, val_dataloader)
    model.load_from_checkpoint(model_checker.best_model_path)   # load best model checkpoint

    # Store prediction from best model
    y_trues, cls, predicts, final_predicts = get_predict(model,
                                                         (train_dataloader, val_dataloader))
    df = pd.DataFrame(np.array([y_trues, cls, predicts, final_predicts]).T,
                      columns=['y_true', 'cls', 'rgs', 'final'])
    df.to_csv(f'./predicts/{MODEL_NAME}.csv', index=False)


