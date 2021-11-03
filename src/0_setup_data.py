#########################################################################################################################
# For using TESS dataset,                                                                                               #
# Download TESS Dataset from  https://dataverse.scholarsportal.info/dataset.xhtml?persistentId=doi:10.5683/SP2/E8H2MF   #
# Unzip and place the whole folder under data/raw/TESS/                                                                 #                                                      #
#########################################################################################################################

from scipy.io import wavfile
from glob import glob
import os
import numpy as np
import pandas as pd
import random
import pathlib
import librosa

from definition import DatasetDefinition

def determine_class(dataset, filename):
    if dataset=="TESS":
        dict_class = {'angry':0,'disgust':1,'fear':2,'happy':3,'neutral':4,'ps':5,'sad':6}
        num_class = dict_class[os.path.basename(filename).split('_')[-1].split('.')[0]]
    
    return num_class

def setup_dataset(dataset="TESS"):
    data_def = DatasetDefinition(dataset,0)
    audio_files = glob(os.path.join(data_def.raw_dataset_path, "**","*.wav"), recursive=True)
    p = pathlib.Path(data_def.interim_audio_path)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    
    p = pathlib.Path(data_def.processed_audio_path)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    
    class_counts =np.zeros((data_def.n_classes,))
    for i in range(len(audio_files)):
        print("Now Processing "+ audio_files[i],end="\r")
        num_class = determine_class(dataset, audio_files[i])
        if num_class > -1:
            audio, _ = librosa.load(audio_files[i], sr=data_def.target_sr, mono=True)
            wavfile.write(os.path.join(data_def.interim_audio_path, str(num_class)+'_'+os.path.basename(audio_files[i])), 
                          data_def.target_sr, audio)
            class_counts[num_class]=class_counts[num_class]+1

    print("class counts" + str(class_counts))

    # generate 10-fold split goups
    audio_files = glob(os.path.join(data_def.interim_audio_path, "*.wav"), recursive=True)
    audio_files_basename = [os.path.basename(x).split('.')[0] +'.pkl' for x in audio_files]
    split_groups = pd.DataFrame(index=audio_files_basename, columns=['group_number'])
    n_per_group = int(data_def.n_per_class/data_def.n_data_split_groups)
    for i in range(data_def.n_classes):
        shuffled = random.sample(audio_files_basename[data_def.n_per_class*i:data_def.n_per_class*(i+1)], data_def.n_per_class)
        for j in range(data_def.n_data_split_groups):
            for k in range(n_per_group):
                split_groups.at[shuffled[j*n_per_group+k],'group_number'] = j

    split_groups.to_csv(data_def.interim_data_split_list)
    split_groups.to_csv(data_def.processed_data_split_list)


if __name__ == '__main__':
    import sys
    args=sys.argv
    dataset = args[1]
    setup_dataset(dataset)
    

