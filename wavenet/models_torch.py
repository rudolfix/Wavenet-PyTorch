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
        self.h_class = Conv1d(num_hidden, num_classes, 1)

    def forward(self, x):
        return self.h_class(self.hs(x))

    def train(self, dataloader, device=None, num_epochs=25, validation=False):
        since = time.time()
        
        best_model_wts = copy.deepcopy(self.state_dict())
        best_acc = 0.0
        
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        if validation:
            phase = 'Validation'
        else:
            phase = 'Training'

        retain = True
        for epoch in range(num_epochs):
            # display epoch
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
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                self.optimizer.zero_grad()
                
                # track history only during training phase
                with torch.set_grad_enabled(not validation):
                    outputs = self(inputs)
                    
                    loss = self.criterion(outputs, labels)
                    
                    if not validation:
                        loss.backward()
                        self.optimizer.step()
                        
                running_loss += loss.item() * inputs.size(1)
                
            epoch_loss = running_loss / len(dataloader)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            print()
