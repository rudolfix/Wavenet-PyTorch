import time, copy
from collections import OrderedDict
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
        super().__init__()
        self.num_time_samples = num_time_samples
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.kernel_size = kernel_size
        
        self.set_device()

        hs = OrderedDict()
        first = True
        for b in range(num_blocks):
            for i in range(num_layers):
                rate = 2**i
                name = 'b{}-l{}'.format(b, i)
                if first:
                    h = GatedConv1d(num_channels, num_hidden, kernel_size, dilation=rate)
                    first = False
                else:
                    h = GatedConv1d(num_hidden, num_hidden, kernel_size, dilation=rate)

                hs[name] = h
                hs[name + '-bn'] = nn.BatchNorm1d(num_hidden)
                # hs[name + '-relu'] = nn.ReLU()

        self.hs = nn.Sequential(hs)
        self.h_class = nn.Conv1d(num_hidden, num_classes, 2)

    def forward(self, x):
        return self.h_class(self.hs(x))

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
        conv_f = Conv1d(in_channels, out_channels, kernel_size, stride=stride, 
                        padding=padding, dilation=dilation, groups=groups, 
                        bias=bias)
        conv_g = Conv1d(in_channels, out_channels, kernel_size, stride=stride, 
                        padding=padding, dilation=dilation, groups=groups, 
                        bias=bias)

    def forward(self, x):
        return torch.mul(nn.Tanh(conv_f(x)), nn.Sigmoid(conv_g(x)))

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
