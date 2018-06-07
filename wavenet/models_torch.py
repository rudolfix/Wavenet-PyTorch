import time, copy
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn import Module, Conv1d, Softmax, Sequential

class Model(Module):
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
                    h = Conv1d(num_channels, num_hidden, kernel_size, dilation=rate)
                    first = False
                else:
                    h = Conv1d(num_hidden, num_hidden, kernel_size, dilation=rate)
                hs[name] = h

        self.hs = Sequential(hs)
        self.h_class = Conv1d(num_hidden, num_classes, 2)

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
            if disp_interval is not None and epoch % disp_interval == 0:
                print('Epoch {} / {}'.format(epoch, num_epochs - 1))
                print('-' * 10)
            
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
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
                print()

class Generator(object):
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def _shift_insert(self, x, val):
        x = x.narrow(-1, 1, x.shape[-1] - 1)
        val = val.reshape([1] * len(x.shape))
        return torch.cat([x, self.dataset._to_tensor(val)], -1)

    def _predict_val(self, x):
        y = self.predict(x)
        return self.dataset.label2value(y.argmax(dim=1))[0]

    def predict(self, x):
        return self.model(x)

    def run(self, x, num_samples, disp_interval=None):
        x = self.dataset._to_tensor(self.dataset.preprocess(x))
        x = torch.unsqueeze(x, 0)

        out = np.zeros(num_samples)
        for i in range(num_samples):
            if disp_interval is not None and i % disp_interval == 0:
                print('Sample {} / {}'.format(i, num_samples))

            y_i = self._predict_val(x)
            y_decoded = self.dataset.encoder.decode(y_i)
            out[i] = y_decoded

            # shift sequence and insert generated value
            x = self._shift_insert(x, y_i)

        return out
