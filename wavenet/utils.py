from argparse import ArgumentParser

# import numpy as np
# from scipy.io import wavfile

# def normalize(data):
#     temp = np.float32(data) - np.min(data)
#     out = (temp / np.max(temp) - 0.5) * 2
#     return out


# def make_batch(path):
#     data = wavfile.read(path)[1][:, 0]

#     data_ = normalize(data)
#     # data_f = np.sign(data_) * (np.log(1 + 255*np.abs(data_)) / np.log(1 + 255))

#     bins = np.linspace(-1, 1, 256)
#     # Quantize inputs.
#     inputs = np.digitize(data_[0:-1], bins, right=False) - 1
#     inputs = bins[inputs][None, :, None]

#     # Encode targets as ints.
#     targets = (np.digitize(data_[1::], bins, right=False) - 1)[None, :]
#     return inputs, targets


def set_args():
    parser = ArgumentParser(description='Wavenet demo')
    parser.add_argument('--x_len', type=int, default=2**15, help='length of input')
    parser.add_argument('--num_classes', type=int, default=256, help='number of discrete output levels')
    parser.add_argument('--num_layers', type=int, default=14, help='number of convolutional layers per block')
    parser.add_argument('--num_blocks', type=int, default=2, help='number of repeating convolutional layer blocks')
    parser.add_argument('--num_hidden', type=int, default=128, help='number of neurons per layer')
    parser.add_argument('--kernel_size', type=int, default=2, help='width of convolutional kernel')
    parser.add_argument('--learn_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--step_size', type=int, default=50, help='step size of learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of learning rate scheduler')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers')
    parser.add_argument('--num_epochs', type=int, default=25, help='number of training epochs')
    parser.add_argument('--model_file', type=str, default='model.pt', help='filename of model')

    args = parser.parse_args()
    return args