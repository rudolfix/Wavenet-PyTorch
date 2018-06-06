# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 21:25:25 2017

@author: bricklayer
"""

import numpy as np
from scipy.io import wavfile

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler, BatchSampler

from .muencoder import MuEncoder

class AudioData(Dataset):
    def __init__(self, track_list, x_len, bitrate=16, twos_comp=True, classes=256, store_tracks=False, encoder=None):
        self.data = []
        self.tracks = []
        self.x_len = x_len
        self.y_len = 1
        self.n_channels = 1
        self.n_classes = 256
        self.bitrate = bitrate
        self.datarange = (-2**(bitrate - 1), 2**(bitrate - 1) - 1)
        self.twos_comp = twos_comp

        if encoder is None:
            self.encoder = MuEncoder(self.datarange)

        for track in track_list:
            audio, dtype, sample_rate = self.__load_audio_from_wav(track)

            if store_tracks:
                self.tracks.append({'name': track, 
                                    'audio': audio, 
                                    'sample_rate': sample_rate})

            for i in range(0, len(audio) - x_len - 1, x_len + 1):
                x, y = self.__extract_segment(audio, x_len, 1, start_idx=i)

                # apply mu-law encoding
                x = self.encoder.encode(x)
                y = self.encoder.encode(y)

                # set inputs to discrete values
                x = self.quantize(x)
                y = self.quantize(y, label=True)

                self.data.append({'x': x, 'y': y})

        self.dtype = dtype
        self.sample_rate = sample_rate
    
    def __load_audio_from_wav(self, filename):
        # read audio
        sample_rate, audio = wavfile.read(filename)
        assert(audio.dtype=='int16') # assume audio is int16 for now
        assert(sample_rate==44100) # assume sample_rate is 44100 for now
        dtype = audio.dtype

        # combine channels
        audio = np.array(audio)
        if len(audio.shape) > 1:
            audio = np.mean(audio, 1)

        return audio, dtype, sample_rate
    
    def __extract_segment(self, audio, n_x, n_y, start_idx=None):
        n_samples = audio.shape[0]
        n_points = n_x + n_y
        
        if start_idx is None:
            #   select random index from range(0, n_samples - n_points)
            start_idx = np.random.randint(0, n_samples - n_points, 1)[0]
        
        # extract segment
        x = audio[start_idx:start_idx+n_x]
        y = audio[start_idx+n_x:start_idx+n_x+n_y]
        return x, y
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        #return self.__extract_segment(self.tracks[idx], self.x_len, self.y_len)
        x = torch.tensor(self.data[idx]['x'], dtype=torch.float32)
        y = torch.tensor(self.data[idx]['y'], dtype=torch.long)

        if len(x.shape) < 2:
            x = torch.unsqueeze(x, 0)

        return (x, y)
        
    def convert_to_wav(self, audio):
        norm_factor = 2**self.bitrate/2.0
        offset = (not self.twos_comp)*1.0
        scaled_audio = (audio + offset)*norm_factor
        return scaled_audio

    def quantize(self, x, label=False):
        bins = np.linspace(-1, 1, self.n_classes)
        out = np.digitize(x, bins, right=False) - 1

        if not label:
            out = bins[out]

        return out

    def label2value(self, label):
        return np.linspace(-1, 1, self.n_classes)[label.data.numpy().astype(int)]


class AudioBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, drop_last=True):
        super().__init__(RandomSampler(dataset), batch_size, drop_last)

class AudioLoader(DataLoader):
    def __init__(self, dataset, batch_size=8, drop_last=True, num_workers=1):
        sampler = AudioBatchSampler(dataset, batch_size, drop_last)
        super().__init__(dataset, batch_sampler=sampler, num_workers=num_workers)
