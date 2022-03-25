import h5py, logging, argparse, getpass, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import torch, torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb

from torch.utils.data import DataLoader, Dataset, TensorDataset
from model import BruceModel
from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from datetime import datetime
from weakref import ReferenceType


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
            return torch.tensor(self.inputs[idx: idx + self.seq_len])
        else:
            return torch.tensor(self.inputs[idx: idx + self.seq_len]), \
                   torch.tensor(self.cls_labels[idx: idx + self.seq_len]), \
                   torch.tensor(self.rgs_labels[idx: idx + self.seq_len])


class BruceModelCheckpoint(ModelCheckpoint):
    # def save_checkpoint(self, trainer: pl.Trainer):
    #     super(BruceModelCheckpoint, self).save_checkpoint(trainer)
    #     trainer.model.save_df(trainer.logger)

    def _update_best_and_save(self, current, trainer: pl.Trainer, monitor_candidates):
        super(BruceModelCheckpoint, self)._update_best_and_save(current, trainer, monitor_candidates)
        trainer.model.save_df(trainer.logger, trainer.current_epoch)


def preprocessing_data(arr, normalize=True):
    if normalize:
        # Data normalization
        return (arr - arr.min()) / (arr.max() - arr.min())
    else:
        # Data standardization
        return (arr - arr.mean()) / arr.std()


def get_data(path, no_sample, normalize=True):
    f = h5py.File(path, 'r')
    idx = -1 if no_sample else 10000

    inputs = f.get('inputs')[:idx]
    inputs = preprocessing_data(inputs, normalize)
    cls_label = f.get('cls_label')[:idx]
    deposit_thickness = f.get('deposit_thickness')[:idx]
    inner_diameter = f.get('inner_diameter')[:idx]

    return inputs, cls_label, deposit_thickness, inner_diameter


def get_args():
    model_parser = argparse.ArgumentParser()

    # Model argumentss
    model_parser.add_argument('--logging_level', default='INFO', type=str, help="Set logging level")
    model_parser.add_argument('-bb', '--backbone', default='cnn', type=str, help='Model backbone: cnn, lstm, unet')
    model_parser.add_argument('--train_path', default='train.h5', type=str, help='Train data path')
    model_parser.add_argument('--val_path', default='val.h5', type=str, help='Validation data path')
    model_parser.add_argument('--total_training_step', default=1e6, type=int, help='Training step')
    model_parser.add_argument('--warming_step', default=1e5, type=int, help='Warming step')
    model_parser.add_argument('--scheduler', action='store_true', help='Use Learning Rate Scheduler')
    model_parser.add_argument('--rgs_loss', default='mape', type=str, help="Regression loss, default is MAE")
    model_parser.add_argument('--normalize', default=0, type=int, help='Normalize data if used, otherwise standardize ')
    model_parser.add_argument('--no_sample', action='store_true', help='Sample to test data and model')
    model_parser.add_argument('--hidden_size', default=256, type=int, help='Hidden size')
    model_parser.add_argument('--act', default='tanh', type=str, help='Activation of intermediate layer')
    model_parser.add_argument('-sl', '--seq_len', default=32, type=int, help='Sequence len')
    model_parser.add_argument('--cls_w', default=0.8, type=float, help='Classification weight')
    model_parser.add_argument('-co', '--core_out', default=256, type=int, help='Core output channel')
    model_parser.add_argument('--initializer_range', default=0.02, type=float, help='Initializer range')

    # CNN args
    model_parser.add_argument('-nc', '--num_cnn', default=1, type=int, help='Number of CNN Layer')
    model_parser.add_argument('-ks', '--kernel_size', default=[3], action=ParseAction,
                              help='Kernel size for each CNN layer')
    model_parser.add_argument('-oc', '--output_channel', default=[8], action=ParseAction,
                              help='Output channel for each CNN layer')

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
    dt_trues, id_trues = [], []
    dt_predicts, id_predicts = [], []
    cls = []

    for dataloader in dataloaders:
        for batch in dataloader:
            inputs, cls_labels, dt_labels, id_labels = batch
            cls_out, dt_out, id_out = model(inputs)

            dt_trues.extend(dt_labels.reshape(-1).tolist())
            id_trues.extend(id_labels.reshape(-1).tolist())

            cls.extend(torch.sigmoid(cls_out.reshape(-1)).tolist())
            dt_predicts.extend(dt_out.reshape(-1).tolist())
            id_predicts.extend(id_out.reshape(-1).tolist())

    final_predicts = (np.array(cls) >= 0.5) * np.array(dt_predicts)

    return dt_trues, id_trues, cls, dt_predicts, id_predicts, final_predicts


INIT_LR = 1e-3
EPOCH = 100
SCHEDULER_EPOCH = 20
SCHEDULER_RATE = 0.9
CLASS_W = [{0: 0.85, 1: 0.15}, None]
DATETIME_NOW = datetime.now().strftime('%Y%m%d_%H%M')

if __name__ == '__main__':
    # Get arguments & logger
    args = get_args()
    args.loss_weights = [args.cls_w, 1 - args.cls_w]  # set loss weights
    logging.basicConfig(level=args.logging_level)
    logger = logging.getLogger('model')
    logger.info(args.__dict__)

    # Get data
    train_inputs, train_cls_label, train_deposit_thickness, train_inner_diameter = get_data(args.train_path,
                                                                                            args.no_sample,
                                                                                            args.normalize)
    val_inputs, val_cls_label, val_deposit_thickness, val_inner_diameter = get_data(args.val_path,
                                                                                    args.no_sample,
                                                                                    args.normalize)

    # Create Dataloader

    train_dataset = TensorDataset(torch.FloatTensor(train_inputs),
                                  torch.FloatTensor(train_cls_label),
                                  torch.FloatTensor(train_deposit_thickness),
                                  torch.FloatTensor(train_inner_diameter))
    val_dataset = TensorDataset(torch.FloatTensor(val_inputs),
                                torch.FloatTensor(val_cls_label),
                                torch.FloatTensor(val_deposit_thickness),
                                torch.FloatTensor(val_inner_diameter))
    '''
    train_dataset = BruceDataset(inputs=train_x, cls_labels=train_y_cls, rgs_labels=train_y_rgs)
    '''

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Calculate training steps for learning rate scheduler
    steps_per_epoch = int(train_inputs.shape[0] // args.batch_size + 1)
    args.total_training_step = steps_per_epoch * args.num_epoch

    # Generate model
    MODEL_NAME = f'{args.backbone.upper()}-{DATETIME_NOW}_{getpass.getuser()}'
    wandb_logger = WandbLogger(project='Rocsole_DILI', name=MODEL_NAME, log_model='all')
    model = BruceModel(**args.__dict__)
    logger.info(model)

    # Init Logger
    wandb_logger.watch(model, log='all')

    # Init Callbacks
    profiler = AdvancedProfiler()
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stop_callback = EarlyStopping(monitor='val/rgs_loss',
                                        mode='min',
                                        patience=50,
                                        verbose=True)
    model_checker = BruceModelCheckpoint(monitor='val/rgs_loss',
                                         mode='min',
                                         dirpath='./model_checkpoint/',
                                         filename=MODEL_NAME + '_{epoch:02d}',
                                         verbose=True)

    # Init Pytorch Lightning Profiler
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[lr_monitor, early_stop_callback, model_checker],
        # profiler=profiler,
        gpus=args.gpu,
        log_every_n_steps=args.log_step,
        max_epochs=args.num_epoch,
        deterministic=True,
    )

    if args.find_lr:
        lr_finder = trainer.tuner.lr_find(model, train_dataloader, val_dataloader, min_lr=1e-5, max_lr=0.1, early_stop_threshold=100)
        print(lr_finder.results)

        fig = lr_finder.plot(suggest=True, show=False)
        plt.savefig('lr_finder.png')

        new_lr = lr_finder.suggestion()
        print('Suggested learning rate: ', new_lr)
        model.hparams.lr = new_lr

    # Fit training data
    trainer.fit(model, train_dataloader, val_dataloader)
    model.load_from_checkpoint(model_checker.best_model_path)  # load best model checkpoint

    # Store prediction from best model
    dt_trues, id_trues, cls, dt_predicts, id_predicts, final_predicts = get_predict(model,
                                                                                    (train_dataloader, val_dataloader))
    df = pd.DataFrame(np.array([dt_trues, id_trues, cls, dt_predicts, id_predicts, final_predicts]).T,
                      columns=['dt_trues', 'id_trues', 'cls', 'dt_predicts', 'id_predicts', 'final_predicts'])
    df.to_csv(f'./predicts/{MODEL_NAME}.csv', index=False)

    '''
    wandb.Table.MAX_ROWS = 1000000
    wandb_logger.experiment.log({'predictions': wandb.Table(dataframe=df)})
    '''