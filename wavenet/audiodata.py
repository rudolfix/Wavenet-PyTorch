# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 21:25:25 2017

@author: bricklayer
"""

import numpy as np
from scipy.io import wavfile

import torch
from torch.utils.data import Dataset

class AudioData(Dataset):
    def __init__(self, track_list, x_len, y_len, bitrate=16, twos_comp=True, store_tracks=False):
        self.data = []
        self.tracks = []
        self.x_len = x_len
        self.y_len = y_len
        self.bitrate = bitrate
        self.twos_comp = twos_comp
        for track in track_list:
            audio, sample_rate = self.__load_audio_from_wav(track)

            if store_tracks:
                self.tracks.append({'name': track, 
                                    'audio': audio, 
                                    'sample_rate': sample_rate})

            for i in range(0, len(audio) - x_len - y_len, x_len + y_len):
                x, y = self.__extract_segment(audio, x_len, y_len, start_idx=i)
                self.data.append({'x': x, 'y': y})
        self.sample_rate = sample_rate
    
    def __load_audio_from_wav(self, filename):
        # read audio
        sample_rate, audio = wavfile.read(filename)
        assert(audio.dtype=='int16') # assume audio is int16 for now
        assert(sample_rate==44100) # assume sample_rate is 44100 for now

        # combine channels
        audio = np.array(audio)
        if len(audio.shape) > 1:
            audio = np.mean(audio, 1)
        
        # normalize to [-1, 1]
        max_code = 2**self.bitrate
        norm_factor = max_code/2.0
        offset = (not self.twos_comp)*max_code
        normed_audio = (audio - offset)/norm_factor
        
        return normed_audio, sample_rate
    
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
        y = torch.tensor(self.data[idx]['y'], dtype=torch.float32)
        return (x, y)
        
    def convert_to_wav(self, audio):
        norm_factor = 2**self.bitrate/2.0
        offset = (not self.twos_comp)*1.0
        scaled_audio = (audio + offset)*norm_factor
        return scaled_audio