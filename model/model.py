import torch, torchmetrics, wandb, transformers
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
                nn.Conv1d(1 if i == 0 else kwargs['output_channel'][i - 1],
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
        self.cnn_out = nn.Linear(kwargs['output_channel'][i] * out_c, kwargs['core_out'])
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


class BruceStackedConv(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, kernel_size=3, **kwargs):
        super().__init__()
        self.stacked_conv = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(out_channel, out_channel, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, inputs):
        return self.stacked_conv(inputs)


class BruceDown(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, kernel_size=3, **kwargs):
        super(BruceDown, self).__init__()
        self.maxpool_cnn = nn.Sequential(
            nn.MaxPool1d(2),
            BruceStackedConv(in_channel, out_channel, kernel_size, **kwargs)
        )

    def forward(self, inputs):
        return self.maxpool_cnn(inputs)


class BruceUp(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, kernel_size=3, **kwargs):
        super(BruceUp, self).__init__()
        self.up_cnn = nn.Sequential(
            nn.Upsample(scale_factor=2),
            BruceStackedConv(in_channel, out_channel, kernel_size, **kwargs)
        )

    def forward(self, inputs, hidden_state):
        return self.up_cnn(inputs) + hidden_state


class BruceProcessingModule(nn.Module):
    def __init__(self, num_feature=512, out_channel=64, kernel_size=3, **kwargs):
        super(BruceProcessingModule, self).__init__()
        self.processing_module = nn.Sequential(
            nn.Linear(524, num_feature),
            nn.Conv1d(1, out_channel, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(out_channel, out_channel, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, inputs):
        b, f = inputs.shape
        inputs = inputs.reshape(b, 1, f).float()
        return self.processing_module(inputs)


class BruceUNet(nn.Module):
    def __init__(self, num_feature=512, output_channel=[64, 128], kernel_size=[3, 3], core_out=512, **kwargs):
        super(BruceUNet, self).__init__()
        self.processing_input = BruceProcessingModule(num_feature=core_out,
                                                      out_channel=output_channel[0],
                                                      kernel_size=kernel_size[0])

        # Build Down Blocks
        downs = []
        for i in range(1, len(output_channel)):
            downs.append(BruceDown(
                in_channel=output_channel[i-1],
                out_channel=output_channel[i],
                kernel_size=kernel_size[i]
            ))
        self.down_blocks = nn.ModuleList(downs)

        # Build Up Blocks
        ups = []
        for i in range(len(output_channel)-1, 0, -1):
            ups.append(BruceUp(
                in_channel=output_channel[i],
                out_channel=output_channel[i-1],
                kernel_size=kernel_size[i]
            ))
        self.up_blocks = nn.ModuleList(ups)

        self.unet_out = nn.Sequential(
            nn.Conv1d(output_channel[0], output_channel[0], kernel_size=kernel_size[0], padding='same'),
            nn.BatchNorm1d(output_channel[0]),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=output_channel[0], out_channels=2, kernel_size=1, padding='same'),
            nn.Dropout(0.1),
            nn.Flatten(),
        )

    def forward(self, x: torch.FloatTensor):
        hidden_states = []

        x = self.processing_input(x)
        hidden_states.insert(0, x)

        # Down forward
        for layer in self.down_blocks:
            x = layer(x)
            hidden_states.insert(0, x)

        # Up forward
        hidden_states.pop(0)
        for i, layer in enumerate(self.up_blocks):
            x = layer(x, hidden_states[i])

        return self.unet_out(x)


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
        #self.save_hyperparameters(kwargs)

        # Set core layer
        if kwargs['backbone'] == 'cnn':
            self.core = BruceCNNCell(**kwargs)
        elif kwargs['backbone'] == 'lstm':
            self.core = BruceLSTMMCell(**kwargs)
        elif kwargs['backbone'] == 'unet':
            self.core = BruceUNet(**kwargs)
            self.core_out *= 2

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
        self.dt_out = nn.Linear(self.core_out, 1)
        self.id_out = nn.Linear(self.core_out, 1)

        # Loss function
        self.cls_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.1275]))
        self.rgs_loss_fn = nn.MSELoss() if self.rgs_loss == 'mse' else nn.L1Loss()

        '''
        # Metrics to log
        self.train_acc = torchmetrics.Accuracy()
        self.train_auc = torchmetrics.AUROC(pos_label=1)
        self.val_acc = torchmetrics.Accuracy()
        self.val_auc = torchmetrics.AUROC(pos_label=1)
        '''

        # Init weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

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
        dt_out = self.dt_out(rgs_out)
        id_out = self.id_out(rgs_out)

        return cls_out, dt_out, id_out

    def loss(self, cls_out, dt_out, id_out, cls_labels, dt_labels, id_labels):
        return self.cls_loss_fn(cls_out, cls_labels.float()), self.rgs_loss_fn(dt_out, dt_labels), self.rgs_loss_fn(id_out, id_labels)

    def training_step(self, batch, batch_idx):
        if self.trainer.global_step == 0:
            wandb.define_metric('train/rgs_loss', summary='min', goal='minimize')
        inputs, cls_labels, dt_labels, id_labels = batch
        cls_out, dt_out, id_out = self(inputs)

        cls_loss, rgs_loss, id_loss = self.loss(cls_out, dt_out, id_out, cls_labels, dt_labels, id_labels)
        #loss = cls_loss * self.loss_weights[0] + rgs_loss * self.loss_weights[1]
        loss = (cls_loss + rgs_loss + id_loss) / 3

        # Log loss
        self.log('train/cls_loss', cls_loss.item(), prog_bar=False)
        self.log('train/rgs_loss', rgs_loss.item(), prog_bar=True)
        self.log('train/id_loss', id_loss.item(), prog_bar=False)

        # Log learning rate to progress bar
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", cur_lr, logger=False, prog_bar=True, on_step=True, on_epoch=False)

        '''
        # Calculate train metrics
        self.train_acc(torch.sigmoid(cls_out), cls_labels.long())
        self.train_auc(torch.sigmoid(cls_out), cls_labels.long())
        '''

        # Log train metrics
        self.log('train/loss', loss.item())
        # self.log('train/acc', self.train_acc)
        # self.log('train/auc', self.train_auc)

        return loss

    def validation_step(self, batch, batch_idx):
        # Track best rgs loss
        if self.trainer.global_step == 0:
            wandb.define_metric('val/rgs_loss', summary='min', goal='minimize')
        inputs, cls_labels, dt_labels, id_labels = batch
        cls_out, dt_out, id_out = self(inputs)

        cls_loss, rgs_loss, id_loss = self.loss(cls_out, dt_out, id_out, cls_labels, dt_labels, id_labels)
        # loss = cls_loss * self.loss_weights[0] + rgs_loss * self.loss_weights[1]
        loss = (cls_loss + rgs_loss + id_loss) / 3

        # Log loss
        self.log('val/cls_loss', cls_loss.item(), prog_bar=False)
        self.log('val/rgs_loss', rgs_loss.item(), prog_bar=True)
        self.log('val_rgs_loss', rgs_loss.item(), prog_bar=False, logger=False)
        self.log('val/id_loss', id_loss.item(), prog_bar=False)

        '''
        # Calculate train metrics
        self.val_acc(torch.sigmoid(cls_out), cls_labels.long())
        self.val_auc(torch.sigmoid(cls_out), cls_labels.long())
        '''

        # Log train metrics
        self.log('val/loss', loss.item())
        # self.log('val/acc', self.val_acc, prog_bar=False)
        # self.log('val/auc', self.val_auc, prog_bar=False)

    def configure_optimizers(self):
        def get_lr_scheduler(opt, factor, num_warmup_steps, num_training_steps, last_epoch=-1):
            def lr_lambda(current_step: int):
                if current_step < num_warmup_steps:
                    return max(factor, float(current_step)) / float(max(1, num_warmup_steps))
                return max(
                    factor, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
                )

            return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda, last_epoch)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = get_lr_scheduler(optimizer,
                                     factor=0.05,
                                     num_warmup_steps=self.warming_step,
                                     num_training_steps=self.total_training_step - self.warming_step)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }
