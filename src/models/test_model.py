from tensorflow.keras.optimizers import SGD
import pandas as pd
import os
import numpy as np

import sys
sys.path.append('..')

from definition import DatasetDefinition
from models.load_data import DataLoader
from models.conv_models import ModelConstruction

def model_test_all(model_name='m5', test_group_num=0, dataset="TESS"):
    
    freqs=[500, 550, 600, 650, 700, 750, 1000, 1250, 1500, 1750, 2000, 0]

    model_list = ['lowpass500Hz_100',
                  'lowpass500Hz_050_original_050',
                  'original_100',
                  'original_050_lowpass500Hz_050']

    data_def = DatasetDefinition(dataset, 0)
    data_def.set_test_group(test_group_num)
    respath = os.path.join(data_def.result_model_path, model_name, str(data_def.test_group_num))
    
    model = ModelConstruction(model_name, data_def).model(kernel_size=data_def.kernel_size)
    model.compile(optimizer=SGD(learning_rate=0.001),loss='categorical_crossentropy',metrics=['acc'])
    
    model_list = [os.path.join(respath,x) for x in model_list]
    
    acc_pd=pd.DataFrame(index=freqs)
    loss_pd=pd.DataFrame(index=freqs)

    x_test_all = []
    y_test_all = []
    for i in range(len(freqs)):
        test_def = DatasetDefinition(dataset, freqs[i])
        test_def.set_test_group(test_group_num)
        data_reader = DataLoader(test_def)
        x_test, y_test = data_reader.get_testing_data()
        x_test_all.append(x_test)
        y_test_all.append(y_test)

    
    for i in range(len(model_list)):
        print(model_list[i])
        model.load_weights(model_list[i])
        acc = np.zeros((len(freqs),))
        loss = np.zeros((len(freqs),))
        for j in range(len(freqs)):
            [loss[j], acc[j]] = model.evaluate(x_test_all[j],y_test_all[j],batch_size=32)
        
        acc_pd[os.path.basename(model_list[i]).split('.')[0]] = acc
        loss_pd[os.path.basename(model_list[i]).split('.')[0]] = loss
    
    acc_pd.to_csv(os.path.join(respath,"accuracy_all.csv"))
    loss_pd.to_csv(os.path.join(respath,"loss_all.csv"))


if __name__ == "__main__":
    model_test_all()