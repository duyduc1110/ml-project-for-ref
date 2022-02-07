import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop

from scipy import stats
from scipy import signal

import keras
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, Callback, ModelCheckpoint

import keras.layers.advanced_activations

import h5py
import tensorflow.keras.backend as K

tbCallBack = TensorBoard(log_dir='Graph',
                         histogram_freq=10,
                         write_graph=True,
                         write_images=True)


class Deposit_Metric():
    def __init__(self, y_train_mean, y_train_std):
        self.y_train_mean = y_train_mean
        self.y_train_std = y_train_std

    def mean_depo_acc(self, y_true, y_pred):
        print('shape', K.shape(y_true))
        y_pred = y_pred * self.y_train_std + self.y_train_mean
        y_true = y_true * self.y_train_std + self.y_train_mean
        print('y_true max ', K.max(y_true[:, 2]))
        print('y_true min ', K.min(y_true[:, 2]))
        return K.mean(K.abs(y_pred[:, 2] - y_true[:, 2]))

    def max_depo_acc(self, y_true, y_pred):
        print('shape', K.shape(y_true))
        y_pred = y_pred * self.y_train_std + self.y_train_mean
        y_true = y_true * self.y_train_std + self.y_train_mean
        return K.max(K.abs(y_pred[:, 2] - y_true[:, 2]))


class DepositMetricCallback(Callback):

    def __init__(self, x_valid, y_valid, x_valid_simu, y_valid_simu, y_train_mean, y_train_std):
        # Abstract base class constructor
        super().__init__()
        self.y_train_mean = y_train_mean
        self.y_train_std = y_train_std
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.x_valid_simu = x_valid_simu
        self.y_valid_simu = y_valid_simu

    def on_train_begin(self, logs=None):
        self.valid_mean_depo_acc = []
        self.valid_simu_mean_depo_acc = []
        self.valid_max_depo_acc = []
        self.valid_simu_max_depo_acc = []

    def on_epoch_end(self, epoch, logs=None):
        y_predict_valid = np.asarray(self.model.predict(self.x_valid))
        y_predict_valid_simu = np.asarray(self.model.predict(self.x_valid_simu))

        y_predict_valid = y_predict_valid * self.y_train_std + self.y_train_mean
        y_predict_valid_simu = y_predict_valid_simu * self.y_train_std + self.y_train_mean
        y_valid = self.y_valid * self.y_train_std + self.y_train_mean
        y_valid_simu = self.y_valid_simu * self.y_train_std + self.y_train_mean

        valid_mean_depo_acc = np.mean(np.abs(y_predict_valid[:, 1] - y_valid[:, 1]))
        valid_simu_mean_depo_acc = np.mean(np.abs(y_predict_valid_simu[:, 1] - y_valid_simu[:, 1]))
        valid_max_depo_acc = np.max(np.abs(y_predict_valid[:, 1] - y_valid[:, 1]))
        valid_simu_max_depo_acc = np.max(np.abs(y_predict_valid_simu[:, 1] - y_valid_simu[:, 1]))

        print('valid_mean_depo_acc: ', valid_mean_depo_acc)
        print('valid_simu_mean_depo_acc: ', valid_simu_mean_depo_acc)
        print('valid_max_depo_acc: ', valid_max_depo_acc)
        print('valid_simu_max_depo_acc: ', valid_simu_max_depo_acc)

        self.valid_mean_depo_acc.append(valid_mean_depo_acc)
        self.valid_simu_mean_depo_acc.append(valid_simu_mean_depo_acc)
        self.valid_max_depo_acc.append(valid_max_depo_acc)
        self.valid_simu_max_depo_acc.append(valid_simu_max_depo_acc)


class TestCallback(keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x = self.test_data
        y_predict = self.model.predict(x)
        print('y_predict mean deposit thickness = ', y_predict[:, 2].mean())
        print('y_predict min deposit thickness = ', y_predict[:, 2].min())
        print('y_predict max deposit thickness = ', y_predict[:, 2].max())


def mean_depo_acc(y_true, y_pred):
    # y_pred = y_pred*y_train_std + y_train_mean
    # y_true = y_true*y_train_std + y_train_mean
    print('shape', K.shape(y_true))
    print('y_true max ', K.max(y_true[:, 2]))
    print('y_true min ', K.min(y_true[:, 2]))
    return K.mean(K.abs(y_pred[:, 2] - y_true[:, 2]))


def max_depo_acc(y_true, y_pred):
    return K.max(K.abs(y_pred[:, 2] - y_true[:, 2]))


esCallBack = EarlyStopping(monitor='mae', min_delta=0.0001, patience=200, verbose=1, mode='auto')

filepath = 'nn_train_pos_deposit_new_samples_valid_callback_remove_bad_with_meas_pull_fas.h5'
model_save_CallBack = ModelCheckpoint(filepath, monitor='val_mean_absolute_error', verbose=1, save_best_only=True,
                                      save_weights_only=False, mode='auto', period=10)

batch_size = 128

batch_size_test = 128
epochs = 100

filename = "Samples_PIG_Equinor_final_rad_0_5.mat"

f = h5py.File(filename, 'r')
data = f['Samples'][:]
ind_bad = 1


arr = np.arange(data.shape[0])

ind_sample = np.random.permutation(arr)

N_test = int(1)

N_train = int(data.shape[0] - N_test)

N_sample_with_meas = None
sample_weight = np.array(1 / N_train * np.ones((N_train, 1)))

sample_weight = sample_weight / sample_weight.sum()

x_train = data[ind_sample[0:N_train], 0:524]

x_train = np.delete(x_train, ind_bad, axis=1)

y_train = data[ind_sample[0:N_train], 524:]

x_train_meas_max = np.max(np.abs(x_train))

noise_level = 1e-3

max_meas_train = np.zeros((N_train, 1))

for ii in range(0, N_train):
    max_meas_train[ii] = np.max(np.abs(x_train[ii, :]))
    x_train[ii, :] = x_train[ii, :] + noise_level * max_meas_train[ii] * np.random.randn(1, x_train.shape[1])

x_train_min = x_train.min()
x_train_max = x_train.max()

x_train_mean = np.mean(x_train, axis=0)
x_train_std = np.std(x_train, axis=0)

# x_train = (x_train-x_train_min)/(x_train_max-x_train_min)
x_train = (x_train - x_train_mean) / x_train_std

y_train_min = y_train.min()
y_train_max = y_train.max()

y_train_mean = np.mean(y_train, axis=0)
y_train_std = np.std(y_train, axis=0)

y_train = (y_train - y_train_mean) / y_train_std

# positions
N_output = 8

x_test = data[ind_sample[N_train:], 0:524]
y_test = data[ind_sample[N_train:], 524:]

x_test = np.delete(x_test, ind_bad, axis=1)

noise_level = 0.00

max_meas_test = np.zeros((N_test, 1))

for ii in range(0, N_test):
    max_meas_test[ii] = np.max(np.abs(x_test[ii, :]))
    x_test[ii, :] = x_test[ii, :] + noise_level * max_meas_test[ii] * np.random.randn(1, x_test.shape[1])
    # x_test[ii,:] = x_test[ii,:]  + noise_level*np.multiply(x_test[ii,:],np.random.randn(1,x_test.shape[1]))

x_test_min = x_test.min()
x_test_max = x_test.max()
# x_test = (x_test-x_test_min)/(x_test_max-x_test_min)

x_test = (x_test - x_train_mean) / x_train_std

y_test_min = y_test.min()
y_test_max = y_test.max()

y_test = (y_test - y_train_mean) / y_train_std




# noise included
filename = "Samples_PIG_Equinor_final_independent_test_set_rad_0_5.mat"

f = h5py.File(filename, 'r')

data_valid_simu = f['Samples'][:]

x_valid_simu = data_valid_simu[:, 0:524]

x_valid_simu = np.delete(x_valid_simu, ind_bad, axis=1)

noise_level = 0.001
# noise_level = 0.00
noise_level = 0.01
noise_level = 0.001
# noise_level = 0.01

max_meas_valid_simu = np.zeros((x_valid_simu.shape[0], 1))

for ii in range(0, x_valid_simu.shape[0]):
    max_meas_valid_simu[ii] = np.max(np.abs(x_valid_simu[ii, :]))
    x_valid_simu[ii, :] = x_valid_simu[ii, :] + noise_level * max_meas_valid_simu[ii] * np.random.randn(1,
                                                                                                        x_valid_simu.shape[
                                                                                                            1])

y_valid_simu = data_valid_simu[:, 524:]

x_valid_simu = (x_valid_simu - x_train_mean) / x_train_std

y_valid_simu = (y_valid_simu - y_train_mean) / y_train_std



x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

print('train samples min', x_train.min())
print('train samples max', x_train.max())

print('test samples min', x_test.min())
print('test samples max', x_test.max())

print('valid_simu samples min', x_valid_simu.min())
print('valid_simu samples max', x_valid_simu.max())

# =============================================================================
# 
# =============================================================================


# del model

# N_meas = int(449)
N_meas = int(x_train.shape[1])

deposit_metric = Deposit_Metric(y_train_mean, y_train_std)

outfile = "nn_train_sample_set_mean_std.npz"
np.savez(outfile, x_train_mean=x_train_mean, x_train_std=x_train_std, y_train_mean=y_train_mean,
         y_train_std=y_train_std)

# deposit_metric_callback = DepositMetricCallback(x_valid, y_valid, x_valid_simu, y_valid_simu, y_train_mean, y_train_std)


# N_layers = [3, 5, 7, 9, 10, 12, 14]
N_layers = [3, 5, 7, 9]
# N_layers = [9]

for nn in N_layers:

    filepath = "nn_train_callback"
    filepath += str(nn - 1)
    filepath += '.h5'
    # model_save_CallBack = keras.callbacks.ModelCheckpoint(filepath, monitor='val_mean_absolute_error', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=100)

    # model_save_CallBack = keras.callbacks.ModelCheckpoint(filepath, monitor='val_mean_depo_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=10)
    model_save_CallBack = keras.callbacks.ModelCheckpoint(filepath, monitor='val_mean_absolute_error', verbose=1,
                                                          save_best_only=True, save_weights_only=False, mode='auto',
                                                          period=10)

    # print(nn)

    model = Sequential()

    # N_neurons_in_layers = np.linspace(N_meas,N_output,nn-1,dtype='int32')
    N_neurons_in_layers = np.logspace(np.log10(1.1 * N_meas), np.log10(N_output), nn - 1, dtype='int32')

    for ii in range(1, nn):

        # print(ii)
        # print(N_neurons_in_layers[ii-1])

        if ii == 1:

            act1 = keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
                                      shared_axes=None)
            model.add(Dense(units=N_neurons_in_layers[ii - 1], input_dim=N_meas))
            # 5% of random channels not working
            # model.add(Dropout(0.05))
            model.add(BatchNormalization())
            model.add(act1)

        elif ii == (nn - 1):

            model.add(Dense(N_output))

        else:

            act1 = keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
                                      shared_axes=None)
            model.add(Dense(units=N_neurons_in_layers[ii - 1]))
            model.add(BatchNormalization())
            model.add(act1)

    model.summary()

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

    print('Layers nn = ', nn)
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=2,
                        callbacks=[esCallBack, model_save_CallBack],
                        # sample_weight = sample_weight.flatten(),
                        sample_weight=None,
                        validation_data=(x_valid_simu, y_valid_simu))
    # validation_split = 0.1)

    y_train = y_train * y_train_std + y_train_mean
    y_test = y_test * y_train_std + y_train_mean
    y_valid_simu = y_valid_simu * y_train_std + y_train_mean

    y_predict_valid_simu = model.predict(x_valid_simu)
    y_predict_valid_simu = y_predict_valid_simu * y_train_std + y_train_mean

    y_train_predict = model.predict(x_train)
    y_train_predict = y_train_predict * y_train_std + y_train_mean

    diff = np.abs(y_train_predict - y_train)

    #    y_predict_real_meas = model.predict(x_real_meas)
    #    y_predict_real_meas = y_predict_real_meas*y_train_std + y_train_mean

    print('------------------------------------')
    # diff_gamma_out = diff[:,0]
    diff_gamma_in = diff[:, 0]
    diff_gamma_out = diff[:, 1]
    diff_deposit_thickness = diff[:, 2]

    diff_rad_id = diff[:, 7]

    diff_pos1 = np.sqrt(np.sum(np.power(y_train_predict[:, 3:5] - y_train[:, 3:5], 2), axis=1))
    diff_pos2 = np.sqrt(np.sum(np.power(y_train_predict[:, 5:7] - y_train[:, 5:7], 2), axis=1))

    model_name = "deposit_R2_layers_"
    model_name += str(nn - 1)
    model_name += '.png'

    x = y_train[:, 2];
    y = y_train_predict[:, 2]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    plt.plot(x, y, 'o', label='data', markersize=2)
    # plt.legend()
    plt.title("$R^2$ = {0:.2f}".format(r_value ** 2))
    plt.xlabel('Ground truth')
    plt.ylabel('Prediction')
    plt.savefig(model_name)
    plt.show()
    plt.clf()

    model_name = "gamma_out_R2_layers_"
    model_name += str(nn - 1)
    model_name += '.png'

    x = y_train[:, 1];
    y = y_train_predict[:, 1]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    plt.plot(x, y, 'o', label='data', markersize=2)
    # plt.legend()
    plt.title("$R^2$ = {0:.2f}".format(r_value ** 2))
    plt.xlabel('Ground truth')
    plt.ylabel('Prediction')
    plt.savefig(model_name)
    plt.show()
    plt.clf()

    model_name = "gamma_in_R2_layers_"
    model_name += str(nn - 1)
    model_name += '.png'

    x = y_train[:, 0];
    y = y_train_predict[:, 0]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    plt.plot(x, y, 'o', label='data', markersize=2)
    # plt.legend()
    plt.title("$R^2$ = {0:.2f}".format(r_value ** 2))
    plt.xlabel('Ground truth')
    plt.ylabel('Prediction')
    plt.savefig(model_name)
    plt.show()
    plt.clf()

    model_name = "pos_x1_R2_layers_"
    model_name += str(nn - 1)
    model_name += '.png'

    x = y_train[:, 3];
    y = y_train_predict[:, 3]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    plt.plot(x, y, 'o', label='data', markersize=2)
    # plt.legend()
    plt.title("$R^2$ = {0:.2f}".format(r_value ** 2))
    plt.xlabel('Ground truth')
    plt.ylabel('Prediction')
    plt.savefig(model_name)
    plt.show()
    plt.clf()

    model_name = "pos_y1_R2_layers_"
    model_name += str(nn - 1)
    model_name += '.png'

    x = y_train[:, 4];
    y = y_train_predict[:, 4]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    plt.plot(x, y, 'o', label='data', markersize=2)
    # plt.legend()
    plt.title("$R^2$ = {0:.2f}".format(r_value ** 2))
    plt.xlabel('Ground truth')
    plt.ylabel('Prediction')
    plt.savefig(model_name)
    plt.show()
    plt.clf()

    model_name = "pos_x2_R2_layers_"
    model_name += str(nn - 1)
    model_name += '.png'

    x = y_train[:, 5];
    y = y_train_predict[:, 5]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    plt.plot(x, y, 'o', label='data', markersize=2)
    # plt.legend()
    plt.title("$R^2$ = {0:.2f}".format(r_value ** 2))
    plt.xlabel('Ground truth')
    plt.ylabel('Prediction')
    plt.savefig(model_name)
    plt.show()
    plt.clf()

    model_name = "pos_y2_R2_layers_"
    model_name += str(nn - 1)
    model_name += '.png'

    x = y_train[:, 6];
    y = y_train_predict[:, 6]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    plt.plot(x, y, 'o', label='data', markersize=2)
    # plt.legend()
    plt.title("$R^2$ = {0:.2f}".format(r_value ** 2))
    plt.xlabel('Ground truth')
    plt.ylabel('Prediction')
    plt.savefig(model_name)
    plt.show()
    plt.clf()

    model_name = "rad_id_R2_layers_"
    model_name += str(nn - 1)
    model_name += '.png'

    x = y_train[:, 7];
    y = y_train_predict[:, 7]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    plt.plot(x, y, 'o', label='data', markersize=2)
    # plt.legend()
    plt.title("$R^2$ = {0:.2f}".format(r_value ** 2))
    plt.xlabel('Ground truth')
    plt.ylabel('Prediction')
    plt.savefig(model_name)
    plt.show()
    plt.clf()

    print('----------------------------------------')
    print('Training errors:')
    print('mean absolute error pos1 = ', diff_pos1.mean())
    print('mean absolute error pos2 = ', diff_pos2.mean())

    print('max absolute error pos1 = ', diff_pos1.max())
    print('max absolute error pos2 = ', diff_pos2.max())

    print('mean absolute error gamma out = ', diff_gamma_out.mean())
    print('mean absolute error gamma in = ', diff_gamma_in.mean())
    print('max absolute error gamma out = ', diff_gamma_out.max())
    print('max absolute error gamma in = ', diff_gamma_in.max())
    print('train: mean absolute error deposit thickness = ', diff_deposit_thickness.mean())
    print('train: max absolute error deposit thickness = ', diff_deposit_thickness.max())

    print('train: mean absolute error rad id = ', diff_rad_id.mean())
    print('train: max absolute error rad id = ', diff_rad_id.max())

    #################################################33

    diff = np.abs(y_predict_valid_simu - y_valid_simu)

    print('------------------------------------')
    diff_gamma_out = diff[:, 1]
    diff_gamma_in = diff[:, 0]
    diff_deposit_thickness = diff[:, 2]

    diff_rad_id = diff[:, 7]

    diff_pos1 = np.sqrt(np.sum(np.power(y_predict_valid_simu[:, 3:5] - y_valid_simu[:, 3:5], 2), axis=1))
    diff_pos2 = np.sqrt(np.sum(np.power(y_predict_valid_simu[:, 5:7] - y_valid_simu[:, 5:7], 2), axis=1))

    # print('mean absolute error = ', score[2])

    model_name = "valid_simu_deposit_"
    model_name += str(nn - 1)
    model_name += '.png'

    x = y_valid_simu[:, 2];
    y = y_predict_valid_simu[:, 2]
    plt.plot(x)
    plt.plot(y)
    # plt.legend()
    plt.savefig(model_name)
    plt.show()
    plt.clf()

    model_name = "valid_simu_pos_x1_"
    model_name += str(nn - 1)
    model_name += '.png'

    x = y_valid_simu[:, 3];
    y = y_predict_valid_simu[:, 3]
    plt.plot(x)
    plt.plot(y)
    # plt.legend()
    plt.savefig(model_name)
    plt.show()
    plt.clf()

    model_name = "valid_simu_pos_y1_"
    model_name += str(nn - 1)
    model_name += '.png'

    x = y_valid_simu[:, 4];
    y = y_predict_valid_simu[:, 4]
    plt.plot(x)
    plt.plot(y)
    # plt.legend()
    plt.savefig(model_name)
    plt.show()
    plt.clf()

    model_name = "valid_simu_pos_x2_"
    model_name += str(nn - 1)
    model_name += '.png'

    x = y_valid_simu[:, 5];
    y = y_predict_valid_simu[:, 5]
    plt.plot(x)
    plt.plot(y)
    # plt.legend()
    plt.savefig(model_name)
    plt.show()
    plt.clf()

    model_name = "valid_simu_pos_y2_"
    model_name += str(nn - 1)
    model_name += '.png'

    x = y_valid_simu[:, 6];
    y = y_predict_valid_simu[:, 6]
    plt.plot(x)
    plt.plot(y)
    # plt.legend()
    plt.savefig(model_name)
    plt.show()
    plt.clf()

    model_name = "valid_simu_rad_"
    model_name += str(nn - 1)
    model_name += '.png'

    x = y_valid_simu[:, 7];
    y = y_predict_valid_simu[:, 7]
    plt.plot(x)
    plt.plot(y)
    # plt.legend()
    plt.savefig(model_name)
    plt.show()
    plt.clf()

    model_name = "valid_simu_deposit_R2_layers_"
    model_name += str(nn - 1)
    model_name += '.png'

    x = y_valid_simu[:, 2];
    y = y_predict_valid_simu[:, 2]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    plt.plot(x, y, 'o', label='data', markersize=2)
    # plt.legend()
    plt.title("$R^2$ = {0:.2f}".format(r_value ** 2))
    plt.xlabel('Ground truth')
    plt.ylabel('Prediction')
    plt.savefig(model_name)
    plt.show()
    plt.clf()

    model_name = "valid_simu_gamma_out_R2_layers_"
    model_name += str(nn - 1)
    model_name += '.png'

    x = y_valid_simu[:, 1];
    y = y_predict_valid_simu[:, 1]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    plt.plot(x, y, 'o', label='data', markersize=2)
    # plt.legend()
    plt.title("$R^2$ = {0:.2f}".format(r_value ** 2))
    plt.xlabel('Ground truth')
    plt.ylabel('Prediction')
    plt.savefig(model_name)
    plt.show()
    plt.clf()

    model_name = "valid_simu_gamma_in_R2_layers_"
    model_name += str(nn - 1)
    model_name += '.png'

    x = y_valid_simu[:, 0];
    y = y_predict_valid_simu[:, 0]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    plt.plot(x, y, 'o', label='data', markersize=2)
    # plt.legend()
    plt.title("$R^2$ = {0:.2f}".format(r_value ** 2))
    plt.xlabel('Ground truth')
    plt.ylabel('Prediction')
    plt.savefig(model_name)
    plt.show()
    plt.clf()

    model_name = "valid_simu_pos_x1_R2_layers_"
    model_name += str(nn - 1)
    model_name += '.png'

    x = y_valid_simu[:, 3];
    y = y_predict_valid_simu[:, 3]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    plt.plot(x, y, 'o', label='data', markersize=2)
    # plt.legend()
    plt.title("$R^2$ = {0:.2f}".format(r_value ** 2))
    plt.xlabel('Ground truth')
    plt.ylabel('Prediction')
    plt.savefig(model_name)
    plt.show()
    plt.clf()

    model_name = "valid_simu_pos_y1_R2_layers_"
    model_name += str(nn - 1)
    model_name += '.png'

    x = y_valid_simu[:, 4];
    y = y_predict_valid_simu[:, 4]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    plt.plot(x, y, 'o', label='data', markersize=2)
    # plt.legend()
    plt.title("$R^2$ = {0:.2f}".format(r_value ** 2))
    plt.xlabel('Ground truth')
    plt.ylabel('Prediction')
    plt.savefig(model_name)
    plt.show()
    plt.clf()

    model_name = "valid_simu_pos_x2_R2_layers_"
    model_name += str(nn - 1)
    model_name += '.png'

    x = y_valid_simu[:, 5];
    y = y_predict_valid_simu[:, 5]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    plt.plot(x, y, 'o', label='data', markersize=2)
    # plt.legend()
    plt.title("$R^2$ = {0:.2f}".format(r_value ** 2))
    plt.xlabel('Ground truth')
    plt.ylabel('Prediction')
    plt.savefig(model_name)
    plt.show()
    plt.clf()

    model_name = "valid_simu_pos_y2_R2_layers_"
    model_name += str(nn - 1)
    model_name += '.png'

    x = y_valid_simu[:, 6];
    y = y_predict_valid_simu[:, 6]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    plt.plot(x, y, 'o', label='data', markersize=2)
    # plt.legend()
    plt.title("$R^2$ = {0:.2f}".format(r_value ** 2))
    plt.xlabel('Ground truth')
    plt.ylabel('Prediction')
    plt.savefig(model_name)
    plt.show()
    plt.clf()

    model_name = "valid_simu_rad_id_R2_layers_"
    model_name += str(nn - 1)
    model_name += '.png'

    x = y_valid_simu[:, 7];
    y = y_predict_valid_simu[:, 7]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    plt.plot(x, y, 'o', label='data', markersize=2)
    # plt.legend()
    plt.title("$R^2$ = {0:.2f}".format(r_value ** 2))
    plt.xlabel('Ground truth')
    plt.ylabel('Prediction')
    plt.savefig(model_name)
    plt.show()
    plt.clf()

    model_name = "valid_simu_rad_id_R2_layers_clean_pipes_"
    model_name += str(nn - 1)
    model_name += '.png'

    x = y_valid_simu[y_valid_simu[:, 2] < 0.001, 7]
    y = y_predict_valid_simu[y_valid_simu[:, 2] < 0.001, 7]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    plt.plot(x, y, 'o', label='data', markersize=2)
    # plt.legend()
    plt.title("$R^2$ = {0:.2f}".format(r_value ** 2))
    plt.xlabel('Ground truth')
    plt.ylabel('Prediction')
    plt.savefig(model_name)
    plt.show()
    plt.clf()

    model_name = "valid_simu_rad_id_R2_layers_wax_pipes_"
    model_name += str(nn - 1)
    model_name += '.png'

    x = y_valid_simu[y_valid_simu[:, 2] > 0.001, 7]
    y = y_predict_valid_simu[y_valid_simu[:, 2] > 0.001, 7]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    plt.plot(x, y, 'o', label='data', markersize=2)
    # plt.legend()
    plt.title("$R^2$ = {0:.2f}".format(r_value ** 2))
    plt.xlabel('Ground truth')
    plt.ylabel('Prediction')
    plt.savefig(model_name)
    plt.show()
    plt.clf()

    #    y_save2 = y_train
    #    filename = "y_train"
    #    filename += str(nn-1)
    #    filename += '.txt'
    #    np.savetxt(filename,y_save2)
    #
    #
    #    y_save2 = y_train_predict
    #    filename = "y_train_predict"
    #    filename += str(nn-1)
    #    filename += '.txt'
    #    np.savetxt(filename,y_save2)

    print('----------------------------------------')
    print('Validation errors:')
    print('mean absolute error pos1 = ', diff_pos1.mean())
    print('mean absolute error pos2 = ', diff_pos2.mean())

    print('max absolute error pos1 = ', diff_pos1.max())
    print('max absolute error pos2 = ', diff_pos2.max())

    print('mean absolute error gamma out = ', diff_gamma_out.mean())
    print('mean absolute error gamma in = ', diff_gamma_in.mean())
    print('max absolute error gamma out = ', diff_gamma_out.max())
    print('max absolute error gamma in = ', diff_gamma_in.max())
    print('valid: mean absolute error deposit thickness = ', diff_deposit_thickness.mean())
    print('valid: max absolute error deposit thickness = ', diff_deposit_thickness.max())

    print('valid: mean absolute error rad is = ', diff_rad_id.mean())
    print('valid: max absolute error rad id = ', diff_rad_id.max())

    #################################################

    model_name = "nn_train_pos_deposit_new_samples_valid2_remove_bad_layers_"
    model_name += str(nn - 1)
    model_name += '.png'

    y_save2 = history.history['mean_absolute_error']
    filename = "mean_abs_error"
    filename += str(nn - 1)
    filename += '.txt'
    np.savetxt(filename, y_save2)

    y_save2 = history.history['val_mean_absolute_error']
    filename = "val_mean_abs_error"
    filename += str(nn - 1)
    filename += '.txt'
    np.savetxt(filename, y_save2)

    plt.plot((history.history['mean_absolute_error']), 'r')
    plt.plot((history.history['val_mean_absolute_error']), 'b')
    plt.savefig(model_name)
    plt.show()
    plt.clf()

    #    fig_name = "nn_train_valid_mean_acc_"
    #    fig_name += str(nn-1)
    #    fig_name += '.png'
    #
    #    plt.plot(deposit_metric_callback.valid_mean_depo_acc,'r')
    #    plt.plot(deposit_metric_callback.valid_simu_mean_depo_acc,'b')
    #    plt.legend('Pori valid.','Simu valid.')
    #    plt.title("Mean valid. deposit estimation accuracy")
    #    plt.savefig(fig_name)
    #    plt.show()
    #    plt.clf()
    #
    #    fig_name = "nn_train_valid_max_acc_"
    #    fig_name += str(nn-1)
    #    fig_name += '.png'
    #
    #    plt.plot(deposit_metric_callback.valid_max_depo_acc,'r')
    #    plt.plot(deposit_metric_callback.valid_simu_max_depo_acc,'b')
    #    plt.legend('Pori valid.','Simu valid.')
    #    plt.title("Max valid. deposit estimation accuracy")
    #    plt.savefig(fig_name)
    #    plt.show()
    #    plt.clf()

    y_train = (y_train - y_train_mean) / y_train_std
    y_test = (y_test - y_train_mean) / y_train_std
    y_valid_simu = (y_valid_simu - y_train_mean) / y_train_std

    model_name = "nn_train_"
    model_name += str(nn - 1)
    model_name += '.h5'

    model.save(model_name)  # creates a HDF5 file 'my_model.h5'

    filename_weights = "nn_train_"
    filename_weights += str(nn - 1)
    filename_weights += '.npy'
    nn_weights = model.get_weights()
    np.save(filename_weights, nn_weights, allow_pickle=True)

# plt.plot(y_train[ind,2],y_train[ind,3],'o')
# plt.savefig("Sample error positions.png")
