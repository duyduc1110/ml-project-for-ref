import torch, torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from collections import OrderedDict


class BruceCNNCell(nn.Module):
    def __init__(self, **kwargs):
        super(BruceCNNCell, self).__init__()

        assert kwargs['num_cnn'] <= len(kwargs['kernel_size'])

        cnn_stack = OrderedDict()
        out_c = 524
        for i in range(kwargs['num_cnn']):
            cnn_layer = nn.Sequential(
                nn.Conv1d(1 if i==0 else kwargs['output_channel'][i-1],
                          kwargs['output_channel'][i],
                          kwargs['kernel_size'][i],
                          padding='same'),
                nn.LayerNorm(out_c),
                nn.MaxPool1d(2),
                nn.Dropout(0.1),
            )
            out_c = out_c // 2
            cnn_stack[f'cnn_{i}'] = cnn_layer
        self.cnn_layers = nn.Sequential(cnn_stack)

        self.flatten = nn.Flatten()
        self.cnn_out = nn.Linear(kwargs['output_channel'][-1] * out_c, kwargs['core_out'])
        self.act = nn.GELU()
        self.dropout_cnn = nn.Dropout(0.1)

    def forward(self, inputs):
        b, f = inputs.shape
        x = inputs.reshape(b, 1, f).float()

        for layer in self.cnn_layers:
            x = layer(x)

        x = self.flatten(x)
        x = self.act(self.cnn_out(x))
        x = self.dropout_cnn(x)

        return x


class BruceLSTMMCell(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.lstm = nn.LSTM(input_size=524,
                            hidden_size=args.hidden_size,
                            batch_first=True,
                            num_layers=args.num_lstm_layer,
                            dropout=0.1,
                            bidirectional=args.bi_di,
                            )
        self.norm = nn.LayerNorm(args.hidden_size)

    def forward(self, inputs):
        x = self.lstm(inputs)
        x = self.norm(x)
        return x


class BruceModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.save_hyperparameters()

        # Set core layer
        if kwargs['backbone'] == 'cnn':
            self.core = BruceCNNCell(**kwargs)
        elif kwargs['backbone'] == 'lstm':
            self.core = BruceLSTMMCell(**kwargs)

        # FCN Layer
        self.fc1 = nn.Linear(self.core_out, self.core_out * 4)
        self.fc1_act = nn.GELU()
        self.fc2 = nn.Linear(self.core_out * 4, self.core_out)
        self.fc2_act = nn.GELU()
        self.layernorm_fcn = nn.LayerNorm(self.core_out)
        self.drop_fcn = nn.Dropout(0.1)

        # Cls output
        self.cls_out = nn.Linear(self.core_out, 1)

        # Regression output
        self.cls_embed = nn.Embedding(2, self.core_out, padding_idx=0)
        self.layernorm_rgs = nn.LayerNorm(self.core_out)
        self.rgs_out = nn.Linear(self.core_out, 1)

        # Loss function
        self.cls_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.1275]))
        self.rgs_loss_fn = nn.MSELoss() if self.rgs_loss == 'mse' else nn.L1Loss()

        # Metrics to log
        self.train_acc = torchmetrics.Accuracy()
        self.train_auc = torchmetrics.AUROC(pos_label=1)
        self.val_acc = torchmetrics.Accuracy()
        self.val_auc = torchmetrics.AUROC(pos_label=1)

    def forward(self, inputs):
        b, f = inputs.shape

        # Core forward
        x = self.core(inputs)

        # FCN
        t = self.fc1_act(self.fc1(x))
        t = self.fc2_act(self.fc2(t))
        x = self.layernorm_fcn(x + t)
        x = self.drop_fcn(x)

        # Classification output
        cls_out = self.cls_out(x)

        # Regression output
        bi_cls = (cls_out > 0.5).squeeze().long()
        cls_embed = self.cls_embed(bi_cls)
        rgs_out = self.layernorm_rgs(cls_out + cls_embed)
        rgs_out = self.rgs_out(rgs_out)

        return cls_out, rgs_out

    def loss(self, cls_out, rgs_out, cls_true, rgs_true):
        return self.cls_loss_fn(cls_out, cls_true.float()), self.rgs_loss_fn(rgs_out, rgs_true)

    def training_step(self, batch, batch_idx):
        inputs, cls_labels, rgs_labels = batch
        cls_out, rgs_out = self(inputs)

        cls_loss, rgs_loss = self.loss(cls_out, rgs_out, cls_labels, rgs_labels)
        loss = cls_loss * self.loss_weights[0] + rgs_loss * self.loss_weights[1]

        # Log loss
        self.log('train/cls_loss', cls_loss.item(), prog_bar=True)
        self.log('train/rgs_loss', rgs_loss.item(), prog_bar=True)

        # Calculate train metrics
        self.train_acc(torch.sigmoid(cls_out), cls_labels.long())
        self.train_auc(torch.sigmoid(cls_out), cls_labels.long())

        # Log train metrics
        self.log('train/loss', loss.item())
        self.log('train/acc', self.train_acc)
        self.log('train/auc', self.train_auc)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, cls_labels, rgs_labels = batch
        cls_out, rgs_out = self(inputs)

        cls_loss, rgs_loss = self.loss(cls_out, rgs_out, cls_labels, rgs_labels)
        loss = cls_loss * self.loss_weights[0] + rgs_loss * self.loss_weights[1]

        # Log loss
        self.log('val/cls_loss', cls_loss.item(), prog_bar=True)
        self.log('val/rgs_loss', rgs_loss.item(), prog_bar=True)
        self.log('val_rgs_loss', rgs_loss.item(), prog_bar=False, logger=False)

        # Calculate train metrics
        self.val_acc(torch.sigmoid(cls_out), cls_labels.long())
        self.val_auc(torch.sigmoid(cls_out), cls_labels.long())

        # Log train metrics
        self.log('val/loss', loss)
        self.log('val/acc', self.val_acc, prog_bar=True)
        self.log('val/auc', self.val_auc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
