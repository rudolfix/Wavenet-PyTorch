import os
import torch
from torch import nn, optim

from wavenet.audiodata import AudioData, AudioLoader
from wavenet.models_torch import Model, Generator
from wavenet.utils import set_args


filelist = ['assets/classical.wav']

if __name__ == '__main__':
    args = set_args()

    # create dataset and dataloader
    dataset = AudioData(filelist, args.x_len, num_classes=args.num_classes, 
                        store_tracks=True)
    dataloader = AudioLoader(dataset, batch_size=args.batch_size, 
                             num_workers=args.num_workers)

    # construct and load/train model
    wave_model = Model(args.x_len, num_channels=1, num_classes=args.num_classes, 
                       num_blocks=args.num_blocks, num_layers=args.num_layers,
                       num_hidden=args.num_hidden, kernel_size=args.kernel_size)
    if os.path.isfile(args.model_file):
        print('Loading model data from file: {}'.format(args.model_file))
        wave_model.load_state_dict(torch.load(args.model_file))
    else:
        print('Model data not found: {}'.format(args.model_file))
        print('Training new model.')
        wave_model.criterion = nn.CrossEntropyLoss()
        wave_model.optimizer = optim.Adam(wave_model.parameters(), 
                                          lr=args.learn_rate)
        wave_model.scheduler = optim.lr_scheduler.StepLR(wave_model.optimizer, 
                                                         step_size=args.step_size, 
                                                         gamma=args.gamma)

        wave_model.train(dataloader, disp_interval=1)

        print('Saving model data to file: {}'.format(args.model_file))
        torch.save(wave_model.state_dict(), args.model_file)

    # predict sequence with model
    wave_generator = Generator(wave_model, dataset)
    y = wave_generator.run(dataset.tracks[0]['audio'][:args.x_len], 10)
    print(y)
    print(y.shape)
