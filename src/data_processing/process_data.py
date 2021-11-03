import os
import sys
import pathlib
import pickle
import numpy as np
from glob import glob
from scipy.io import wavfile

sys.path.append("..")

from definition import DatasetDefinition

def read_wav(filename, target_sr):
    _, audio = wavfile.read(filename)
    audio = audio.reshape(-1, 1)
    return audio

def transform_data(dataset, cut_off_freq):
    data_def=DatasetDefinition(dataset, cut_off_freq)
    
    p = pathlib.Path(data_def.processed_audio_path)
    if ~p.exists():
        p.mkdir(parents=True, exist_ok=True)

    wav_file_list = glob(os.path.join(data_def.interim_audio_path, '*.wav'))
     
    for i, wav_filename in enumerate(wav_file_list):
        print("Now processing " + str(i) + '/' + str(len(wav_file_list)), end="\r")
        
        class_id = os.path.basename(wav_filename).split('_')[0]
        audio_data = read_wav(wav_filename, target_sr=data_def.target_sr)
        original_length = len(audio_data)
        
        if original_length < data_def.audio_length:
            audio_data = np.concatenate((audio_data, np.zeros(shape=(data_def.audio_length - original_length, 1))))
        elif original_length > data_def.audio_length:
            audio_data = audio_data[original_length-data_def.audio_length:]

        save_name = os.path.join(data_def.processed_audio_path, os.path.basename(wav_filename).split('.')[0] + '.pkl')
        pickle.dump({'class_id': class_id,'audio': audio_data}, open(save_name, "wb"))


if __name__ == '__main__':
    args=sys.argv
    dataset = args[1]
    if len(args)==2:
        cut_off_freq = 0
    else:
        cut_off_freq = int(args[2])
    
    transform_data(dataset, cut_off_freq)
