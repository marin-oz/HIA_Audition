from scipy.io import wavfile
from glob import glob
import os
import sys
import numpy as np
import pathlib

sys.path.append('..')
from definition import DatasetDefinition


def lowpass_filter(audio,fs,cut_off_freq):
    n = len(audio)  
    dt = 1/fs  
    y = np.reshape(audio,(len(audio,)))
    yf = np.fft.fft(y)/(n/2)
    freq = np.fft.fftfreq(n, dt)
    yf[(freq > cut_off_freq)] = 0
    yf[(freq < 0)] = 0
    y = np.real(np.fft.ifft(yf)*n)
    return  y.astype("float32")

def make_filtered(dataset, cut_off_freq):
    data_def = DatasetDefinition(dataset, cut_off_freq)

    p = pathlib.Path(data_def.interim_audio_path)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    
    for wav_filename in glob(os.path.join(DatasetDefinition(dataset, 0).interim_audio_path, '*.wav')):
        print("Now Processing "+wav_filename,end="\r")
        fs, data = wavfile.read(wav_filename)
        filtered = lowpass_filter(data,fs,cut_off_freq)
        save_name = os.path.join(data_def.interim_audio_path, os.path.basename(wav_filename))
        wavfile.write(save_name, fs, filtered.astype(data.dtype))


if __name__ == '__main__':
    args=sys.argv
    dataset = args[1]
    cut_off_freq=int(args[2])
    make_filtered(dataset, cut_off_freq)