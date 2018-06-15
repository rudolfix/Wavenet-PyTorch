import time, copy
from collections import OrderedDict
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, 
                 num_time_samples,
                 num_channels=1,
                 num_classes=256,
                 num_blocks=2,
                 num_layers=14,
                 num_hidden=128,
                 kernel_size=2):
        super(Model, self).__init__()
        self.num_time_samples = num_time_samples
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.kernel_size = kernel_size
        self.receptive_field = 1 + (kernel_size - 1) * \
                               num_blocks * sum([2**k for k in range(num_layers)])
        self.output_width = num_time_samples - self.receptive_field + 1
        print('receptive_field: {}'.format(self.receptive_field))
        print('Output width: {}'.format(self.output_width))
        
        self.set_device()

        hs = []
        batch_norms = []
        first = True
        for b in range(num_blocks):
            for i in range(num_layers):
                rate = 2**i
                name = 'b{}-l{}'.format(b, i)
                if first:
                    h = GatedResidualBlock(num_channels, num_hidden, kernel_size, self.output_width,
                                           dilation=rate)
                    first = False
                else:
                    h = GatedResidualBlock(num_hidden, num_hidden, kernel_size, self.output_width,
                                           dilation=rate)

                hs.append(h)
                batch_norms.append(nn.BatchNorm1d(num_hidden))

        self.hs = nn.ModuleList(hs)
        self.batch_norms = nn.ModuleList(batch_norms)
        self.relu_1 = nn.ReLU()
        self.conv_1_1 = nn.Conv1d(num_hidden, num_hidden, 1)
        self.relu_2 = nn.ReLU()
        self.conv_1_2 = nn.Conv1d(num_hidden, num_hidden, 1)
        self.h_class = nn.Conv1d(num_hidden, num_classes, 2)

    def forward(self, x):
        skips = []
        for layer, batch_norm in zip(self.hs, self.batch_norms):
            x, skip = layer(x)
            x = batch_norm(x)
            skips.append(skip)

        x = reduce((lambda a, b : torch.add(a, b)), skips)
        x = self.relu_1(self.conv_1_1(x))
        x = self.relu_2(self.conv_1_2(x))
        return self.h_class(x)

    def set_device(self, device=None):
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

    def train(self, dataloader, num_epochs=25, validation=False, disp_interval=None):
        since = time.time()
        
        self.to(self.device)

        if validation:
            phase = 'Validation'
        else:
            phase = 'Training'

        for epoch in range(num_epochs):
            if not validation:
                self.scheduler.step()
                super().train()
            else:
                self.eval()
                
            # reset loss for current phase and epoch
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                # track history only during training phase
                with torch.set_grad_enabled(not validation):
                    outputs = self(inputs)
                    loss = self.criterion(outputs, labels)
                    
                    if not validation:
                        loss.backward()
                        self.optimizer.step()
                        
                running_loss += loss.item() * inputs.size(1)

            if disp_interval is not None and epoch % disp_interval == 0:
                epoch_loss = running_loss / len(dataloader)
                print('Epoch {} / {}'.format(epoch, num_epochs - 1))
                print('Learning Rate: {}'.format(self.scheduler.get_lr()))
                print('{} Loss: {}'.format(phase, epoch_loss))
                print('-' * 10)
                print()

class GatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True):
        super(GatedConv1d, self).__init__()
        self.dilation = dilation
        self.conv_f = nn.Conv1d(in_channels, out_channels, kernel_size, 
                                stride=stride, padding=padding, dilation=dilation, 
                                groups=groups, bias=bias)
        self.conv_g = nn.Conv1d(in_channels, out_channels, kernel_size, 
                                stride=stride, padding=padding, dilation=dilation, 
                                groups=groups, bias=bias)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        padding = self.dilation - (x.shape[-1] + self.dilation - 1) % self.dilation
        x = nn.functional.pad(x, (self.dilation, 0))
        return torch.mul(self.conv_f(x), self.sig(self.conv_g(x)))

class GatedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, output_width, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True):
        super(GatedResidualBlock, self).__init__()
        self.output_width = output_width
        self.gatedconv = GatedConv1d(in_channels, out_channels, kernel_size, 
                                     stride=stride, padding=padding, 
                                     dilation=dilation, groups=groups, bias=bias)
        self.conv_1 = nn.Conv1d(out_channels, out_channels, 1, stride=1, padding=0,
                                dilation=1, groups=1, bias=bias)

    def forward(self, x):
        skip = self.conv_1(self.gatedconv(x))
        residual = torch.add(skip, x)

        skip_cut = skip.shape[-1] - self.output_width
        skip = skip.narrow(-1, skip_cut, self.output_width)
        return residual, skip

class Generator(object):
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def _shift_insert(self, x, val):
        x = x.narrow(-1, 1, x.shape[-1] - 1)
        val = val.reshape([1] * len(x.shape))
        return torch.cat([x, self.dataset._to_tensor(val)], -1)

    def tensor2numpy(self, x):
        return x.data.numpy()

    def predict(self, x):
        x = x.to(self.model.device)
        self.model.to(self.model.device)
        return self.model(x)

    def run(self, x, num_samples, disp_interval=None):
        x = self.dataset._to_tensor(self.dataset.preprocess(x))
        x = torch.unsqueeze(x, 0)

        out = np.zeros(num_samples)
        for i in range(num_samples):
            if disp_interval is not None and i % disp_interval == 0:
                print('Sample {} / {}'.format(i, num_samples))

            y_i = self.tensor2numpy(self.predict(x).cpu())
            y_i = self.dataset.label2value(y_i.argmax(axis=1))[0]
            y_decoded = self.dataset.encoder.decode(y_i)
            out[i] = y_decoded

            # shift sequence and insert generated value
            x = self._shift_insert(x, y_i)

        return out
