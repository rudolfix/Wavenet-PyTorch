from argparse import ArgumentParser
from torch import nn, optim

from wavenet.audiodata import AudioData, AudioLoader
from wavenet.models_torch import Model


filelist = ['assets/classical.wav']

def set_args():
    parser = ArgumentParser(description='Wavenet demo')
    parser.add_argument('--x_len', type=int, default=2**10, help='length of input')
    parser.add_argument('--num_classes', type=int, default=256, help='number of discrete output levels')
    parser.add_argument('--num_layers', type=int, default=9, help='number of convolutional layers per block')
    parser.add_argument('--num_blocks', type=int, default=2, help='number of repeating convolutional layer blocks')
    parser.add_argument('--num_hidden', type=int, default=64, help='number of neurons per layer')
    parser.add_argument('--kernel_size', type=int, default=2, help='width of convolutional kernel')
    parser.add_argument('--learn_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--step_size', type=int, default=50, help='step size of learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of learning rate scheduler')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = set_args()
    dataset = AudioData(filelist, args.x_len, classes=args.num_classes, 
                        store_tracks=True)
    dataloader = AudioLoader(dataset, batch_size=args.batch_size, 
                             num_workers=args.num_workers)

    wave_model = Model(args.x_len, num_channels=1, num_classes=args.num_classes, 
                       num_blocks=args.num_blocks, num_layers=args.num_layers,
                       num_hidden=args.num_hidden, kernel_size=args.kernel_size)

    wave_model.criterion = nn.CrossEntropyLoss()
    wave_model.optimizer = optim.Adam(wave_model.parameters(), 
                                      lr=args.learn_rate)
    wave_model.scheduler = optim.lr_scheduler.StepLR(wave_model.optimizer, 
                                                     step_size=args.step_size, 
                                                     gamma=args.gamma)

    wave_model.train(dataloader)
