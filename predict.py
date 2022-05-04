import h5py, logging, argparse, getpass, pandas as pd, numpy as np
import torch, torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset, TensorDataset

from sklearn.model_selection import train_test_split


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

    train_rgs_labels = data[:, 526].reshape(-1, 1)
    train_cls_labels = (train_rgs_labels > 0).reshape(-1, 1)
    train_inputs = preprocessing_data(data[:, :524], True)

    return train_inputs, train_cls_labels, train_rgs_labels


def get_args():
    model_parser = argparse.ArgumentParser()

    # Model argumentss
    model_parser.add_argument('--logging_level', default='INFO', type=str, help="Set logging level")
    model_parser.add_argument('--model_path', default='model_checkpoint\CNN-20220223_0122_Bonz_epoch=01-val_rgs_loss=0.24.ckpt', type=str, help="Model path")
    model_parser.add_argument('--rgs_loss', default='mae', type=str, help="Regression loss, default is MAE")
    model_parser.add_argument('--data_path', default='data.mat', type=str, help='Data path')
    model_parser.add_argument('--no_sample', action='store_true', help='Sample to test data and model')
    model_parser.add_argument('--bi_di', action='store_true', help='Bi-directional for RNN')
    model_parser.add_argument('--hidden_size', default=256, type=int, help='Hidden size')
    model_parser.add_argument('--num_lstm_layer', default=1, type=int, help='Number of LSTM layer')
    model_parser.add_argument('--cls_w', default=0.8, type=float, help='Number of LSTM layer')

    # Trainer arguments
    model_parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    model_parser.add_argument('--batch_size', default=128, type=int, help='Batch size per device')
    model_parser.add_argument('--log_step', default=100, type=int, help='Steps per log')
    model_parser.add_argument('--gpu', default=0, type=int, help='Use GPUs')
    model_parser.add_argument('--num_epoch', default=10, type=int, help='Number of epoch')

    args = model_parser.parse_args()
    return args


def get_predict(model, dataloader):
    y_trues = []
    predicts = []
    cls = []

    for batch in dataloader:
        inputs, cls_labels, rgs_labels = batch
        cls_out, rgs_out = model(inputs)

        y_trues.extend(rgs_labels.reshape(-1).tolist())
        cls.extend(torch.sigmoid(cls_out.reshape(-1)).tolist())
        predicts.extend(rgs_out.reshape(-1).tolist())

    final_predicts = (np.array(cls) >= 0.5) * np.array(predicts)

    return y_trues, cls, predicts, final_predicts


if __name__ == '__main__':
    # Get arguments & logger
    args = get_args()
    args.loss_weights = [args.cls_w, 1 - args.cls_w] # set loss weights
    logging.basicConfig(level=args.logging_level)
    logger = logging.getLogger('model')

    # Generate model
    model = BruceModel.load_from_checkpoint(args.model_path)

    # Get data
    train_inputs, train_cls_labels, train_rgs_labels = get_data(args.data_path, args.no_sample)

    # Create Dataloader
    train_dataset = TensorDataset(torch.FloatTensor(train_inputs),
                                  torch.FloatTensor(train_cls_labels),
                                  torch.FloatTensor(train_rgs_labels))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)

    y_trues, cls, predicts, final_predicts = get_predict(model, train_dataloader)
    df = pd.DataFrame(np.array([y_trues, cls, predicts, final_predicts]).T, columns=['y_true', 'cls', 'rgs', 'final'])
    df.to_csv(f'./predicts/predicted.csv', index=False)



