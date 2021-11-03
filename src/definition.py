import os
import sys
import pathlib

PROJECT_PATH = pathlib.Path(__file__).resolve().parent.parent

class DatasetDefinition:
    
    test_group_num = 0

    def __init__(self,dataset,cut_off_freq):
        if(dataset=="TESS"):
            self.target_sr = 24414
            self.audio_length = 75000
            self.n_classes = 7
            self.n_data_split_groups = 10
            self.n_per_class = 400
            self.kernel_size = 244
        else:
            sys.exit("ERROR : dataset " + dataset + " does not exxit.")
        
        self.result_model_path = os.path.join(PROJECT_PATH, "trained_models", dataset)
        self.processed_dataset_path = os.path.join(PROJECT_PATH,"data","processed",dataset)
        self.interim_dataset_path = os.path.join(PROJECT_PATH,"data","interim",dataset)
        self.raw_dataset_path = os.path.join(PROJECT_PATH,"data","raw",dataset)
        self.interim_data_split_list = os.path.join(self.interim_dataset_path, "data_split_groups.csv")
        self.processed_data_split_list = os.path.join(self.processed_dataset_path, "data_split_groups.csv")

        if cut_off_freq == 0:
            dirname = "original"
        else:
            dirname = "lowpass_"+str(cut_off_freq)+"Hz"
        
        self.interim_audio_path = os.path.join(self.interim_dataset_path,dirname)
        self.processed_audio_path = os.path.join(self.processed_dataset_path,dirname)

    def set_test_group(self,group_number):
        self.test_group_num = group_number

       