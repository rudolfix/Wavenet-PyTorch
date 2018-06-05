import matplotlib.pyplot as plt
import numpy as np
from torch.nn import Conv1d, Softmax

class Model(object):
    def __init__(self, 
                 num_time_samples,
                 num_channels=1,
                 num_classes=256,
                 num_blocks=2,
                 num_layers=14,
                 num_hidden=128,
                 kernel_size=2):
        
        self.num_time_samples = num_time_samples
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.kernel_size = kernel_size
        self.gpu_fraction = gpu_fraction

        self.hs = []
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
                self.hs.append(h) 

        h_class = Conv1d(num_hidden, num_classes, kernel_size)
        softmax = Softmax()

        def forward(self, x):
            for h in self.hs:
                x = h(x)

            output = softmax(h_class(x))
            return output