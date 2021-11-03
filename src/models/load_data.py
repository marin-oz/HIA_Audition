import pickle
import numpy as np
import pandas as pd
import os

from tensorflow.keras.utils import to_categorical


class DataLoader:
    def __init__(self,data_def):
        split_group = pd.read_csv(data_def.processed_data_split_list,index_col=0)
        print('group number ' + str(data_def.test_group_num) + " is left out for test")

        train_group = list(split_group.index[split_group['group_number']!=data_def.test_group_num])
        self.train_files = [os.path.join(data_def.processed_audio_path,x) for x in train_group]
        print('training files =', len(self.train_files))

        test_group = list(split_group.index[split_group['group_number']==data_def.test_group_num])
        self.test_files = [os.path.join(data_def.processed_audio_path,x) for x in test_group]
        print('testing files =', len(self.test_files))

        self.n_classes = data_def.n_classes
       
    def get_training_data(self):
        print('Loading training data')
        return self._get_data(self.train_files)

    def get_testing_data(self):
        print('Loading testing data')
        return self._get_data(self.test_files)

    def _get_data(self, file_list):
        x, y = [], []
        for i, filename in enumerate(file_list):
            print("Now Loading " + str(i) + '/' + str(len(file_list)), end="\r")
            with open(filename, 'rb') as f:
                audio_file = pickle.load(f)
                x.append(audio_file['audio'])
                y.append(audio_file['class_id'])
        
        x = np.array(x)
        y = to_categorical(np.array(y), num_classes=self.n_classes)
        return x, y



if __name__ == '__main__':
    import sys
    sys.path.append("..")
    from definition import DatasetDefinition

    dpath = DatasetDefinition('TESS',0)
    data_reader = DataLoader(dpath)
