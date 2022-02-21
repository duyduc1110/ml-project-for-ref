import torch, torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class BruceCNNModule(nn.Module):
    def __init__(self):
        super(BruceCNNModule, self).__init__()
        self.cnn1 = nn.Conv1d(1, 8, 8, padding='same')
        self.ln1 = nn.LayerNorm(524)
        self.avgpool1 = nn.AvgPool1d(8)
        self.cnn2 = nn.Conv1d(1, 8, 16, padding='same')
        self.ln2 = nn.LayerNorm(524)
        self.avgpool2 = nn.AvgPool1d(8)
        self.flatten = nn.Flatten()
        self.cnn_out = nn.Linear(1040, 64)
        self.dropout_cnn = nn.Dropout(0.1)

    def forward(self, inputs):
        b, f = inputs.shape
        # CNN forward
        x1 = self.avgpool1(self.ln1(self.cnn1(inputs.reshape(b, 1, f).float())))
        x2 = self.avgpool2(self.ln2(self.cnn2(inputs.reshape(b, 1, f).float())))
        x = torch.cat([x1, x2], -1)
        x = self.flatten(x)
        x = F.gelu(self.cnn_out(x))
        x = self.dropout_cnn(x)

        return x
        

class BruceLSTMModule(nn.Module):
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


class BruceCNNModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        # CNN Layer
        self.cnn1 = nn.Conv1d(1, 8, 8, padding='same')
        self.ln1 = nn.LayerNorm(524)
        self.avgpool1 = nn.AvgPool1d(8)
        self.cnn2 = nn.Conv1d(1, 8, 16, padding='same')
        self.ln2 = nn.LayerNorm(524)
        self.avgpool2 = nn.AvgPool1d(8)
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
        self.rgs_loss_fn = nn.MSELoss()

        # Loss weights
        self.loss_weights = torch.tensor(args.loss_weights).float()

        # Metrics to log
        self.train_acc = torchmetrics.Accuracy()
        self.train_auc = torchmetrics.AUROC(pos_label=1)
        self.val_acc = torchmetrics.Accuracy()
        self.val_auc = torchmetrics.AUROC(pos_label=1)

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
        self.log('train/loss', loss)
        self.log('train/acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/auc', self.train_auc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, cls_labels, rgs_labels = batch
        cls_out, rgs_out = self(inputs)

        cls_loss, rgs_loss = self.loss(cls_out, rgs_out, cls_labels, rgs_labels)
        loss = cls_loss * self.loss_weights[0] + rgs_loss * self.loss_weights[1]

        # Calculate train metrics
        self.val_acc(torch.sigmoid(cls_out), cls_labels.long())
        self.val_auc(torch.sigmoid(cls_out), cls_labels.long())

        # Log train metrics
        self.log('val/loss', loss)
        self.log('val/acc', self.val_acc)
        self.log('val/auc', self.val_auc)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.args.lr)


class BruceRNNModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        # LSTM Layer
        self.lstm = nn.LSTM(input_size=524,
                            hidden_size=args.hidden_size,
                            batch_first=True,
                            dropout=0.1,
                            bidirectional=args.bi_di,
                            )
        self.norm = nn.LayerNorm(args.hidden_size)

        # FCN Layer
        self.fc1 = nn.Linear(64, 512)
        self.fc1_act = nn.GELU()
        self.fc2 = nn.Linear(512, 64)
        self.fc2_act = nn.GELU()
        self.norm = nn.LayerNorm(64)
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
        self.loss_weights = torch.tensor(args.loss_weights).float()

        # Metrics to log
        self.train_acc = torchmetrics.Accuracy()
        self.train_auc = torchmetrics.AUROC(pos_label=1)
        self.val_acc = torchmetrics.Accuracy()
        self.val_auc = torchmetrics.AUROC(pos_label=1)