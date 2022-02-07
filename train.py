import pandas as pd, numpy as np, tensorflow as tf, tensorflow.keras as K
import h5py
import matplotlib.pyplot as plt
import sklearn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

from sklearn.metrics import classification_report


class BruceModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(524, 1)

    def forward(self, inputs):
        return self.fc1(inputs)

    def training_step(self)


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


def scheduler(epoch, lr):
    if epoch % 2 == 0 and epoch >= SCHEDULER_EPOCH:
        return lr * SCHEDULER_RATE
    return lr


def build_model():
    inputs = K.layers.Input(shape=(524, 1), name='inputs')

    # CNN Layer
    x1 = K.layers.Conv1D(10, 8, padding='same')(inputs)
    x1 = K.layers.BatchNormalization(axis=-2)(x1)
    x1 = K.layers.MaxPool2D((5, 2), padding='same')(tf.expand_dims(x1, -1))

    x2 = K.layers.Conv1D(10, 16, padding='same')(inputs)
    x2 = K.layers.BatchNormalization(axis=-2)(x2)
    x2 = K.layers.MaxPool2D((5, 2), padding='same')(tf.expand_dims(x2, -1))

    x = K.layers.concatenate([x1, x2], axis=-2)
    x = K.layers.Flatten()(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dropout(0.1)(x)

    # MLP Layer
    x = K.layers.Dense(256, activation=tf.nn.gelu, name='dense1')(x)
    x = K.layers.Dropout(0.2, name='dropout1')(x)
    x = K.layers.Dense(32, activation=tf.nn.gelu, name='dense2')(x)
    x = K.layers.Dropout(0.2, name='dropout2')(x)

    # Output layers
    cls_out = K.layers.Dense(1, activation=tf.nn.sigmoid, name='cls_output')(x)
    rgs_out = K.layers.Dense(1, activation=tf.nn.sigmoid, name='rgs_output')(x)

    model = K.Model(inputs=inputs, outputs=[cls_out, rgs_out])

    return model


INIT_LR = 1e-3
LOSS_WEIGHTS = [1, 1]
BATCH_SIZE = 2048
EPOCH = 100
SCHEDULER_EPOCH = 20
SCHEDULER_RATE = 0.9
CLASS_W = [{0: 0.85, 1: 0.15}, None]
METRICS = ['accuracy', K.metrics.AUC()]

if __name__ == '__main__':
    f = read_data()
    data = f['Samples_big'][:]

    train_rgs_labels = data[:, 526].reshape(-1, 1)
    train_cls_labels = (train_rgs_labels > 0).reshape(-1, 1)
    train_inputs = preprocessing_data(data[:, :524], True)

    scheduler = K.callbacks.LearningRateScheduler(scheduler)

    model = build_model()
    model.summary()

    CLASS_W = {'cls_output': {0: 0.85, 1: 0.15}}
    model.compile(
        optimizer=K.optimizers.Adam(INIT_LR),
        loss=[K.losses.binary_crossentropy, K.losses.MeanAbsolutePercentageError()],
        loss_weights=LOSS_WEIGHTS,
        metrics=METRICS,
    )

    hist = model.fit(
        x=train_inputs,
        y=[train_cls_labels, train_rgs_labels],
        callbacks=[scheduler],
        batch_size=BATCH_SIZE,
        epochs=EPOCH,
        validation_split=0.3,
        class_weight=CLASS_W
    )

    plt.plot(hist.history['auc'], label='Training AUC')
    plt.plot(hist.history['val_auc'], label='Val AUC')
    plt.legend()
    plt.show()
