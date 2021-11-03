from tensorflow.keras.optimizers import SGD
import pandas as pd
import os
import numpy as np

import sys
sys.path.append('..')

from definition import DatasetDefinition
from models.load_data import DataLoader
from models.conv_models import ModelConstruction


def ablation(model_name='m5', test_group_num=0, dataset="TESS", high_freq_order=True, random_order=False):

    freqs=[0, 500]
    
    if model_name=="m3":
        num_filters=256
    elif model_name=="m5":
        num_filters=128
    else:
        sys.exit("ERROR : model " + model_name + " does not exxit.")
    
    
    data_def = DatasetDefinition(dataset, 0)
    data_def.set_test_group(test_group_num)

    respath = os.path.join(data_def.result_model_path, model_name, str(data_def.test_group_num))

    kernel_size = data_def.kernel_size
    model = ModelConstruction(model_name, data_def).model(kernel_size=kernel_size)
    model.compile(optimizer=SGD(learning_rate=0.001),loss='categorical_crossentropy',metrics=['acc'])
    
    model_list = ['lowpass500Hz_050_original_050',
                  'original_050_lowpass500Hz_050',
                  'lowpass500Hz_100', 
                  'original_100']

    x_test_all = []
    y_test_all = []
    for i in range(len(freqs)):
        test_def = DatasetDefinition(dataset, freqs[i])
        test_def.set_test_group(test_group_num)
        data_reader = DataLoader(test_def)
        x_test, y_test = data_reader.get_testing_data()
        x_test_all.append(x_test)
        y_test_all.append(y_test)
    
    result_pd = pd.DataFrame()

    for trained_model in model_list:
        model.load_weights(os.path.join(respath,trained_model))
        result = np.zeros((num_filters+1,len(freqs)))

        for i in range(len(freqs)):
            [loss,result[0,i]] = model.evaluate(x_test_all[i],y_test_all[i],batch_size=10)

        w = model.layers[0].get_weights()
        w1= w[0]

        filters=np.zeros((num_filters,kernel_size//4-1))
        for i in range(num_filters):
            amp=np.abs((np.fft.fft(w1[:,0,i])))**2
            filters[i,:]=amp[1:kernel_size//4]/sum(amp[1:kernel_size//4])
        vec = np.lexsort((1/np.max(filters,axis=1),np.argmax(filters,axis=1)))

        if high_freq_order:
            vec = np.flip(vec)
        if random_order:
            vec = range(num_filters)
        
        for i in range(num_filters):
            w[0][:,:,vec[i]] = 0 
            w[1][vec[i]] = 0
            model.layers[0].set_weights(w)

            for j in range(len(freqs)):
                [loss,result[i+1,j]] = model.evaluate(x_test_all[j],y_test_all[j],batch_size=10)

        for i in range(len(freqs)):
            if freqs[i]==0:
                f = 'original'
            else:
                f = "lowpass_"+str(freqs[i])+"Hz"
            
            result_pd[trained_model+'_test_on_' + f] = result[:,i]

    if random_order:
        result_pd.to_csv(os.path.join(respath,'fourier_filter_ablation_random.csv'))
    elif high_freq_order:
        result_pd.to_csv(os.path.join(respath,'fourier_filter_ablation_high_freq.csv'))
    else:
        result_pd.to_csv(os.path.join(respath,'fourier_filter_ablation_low_freq.csv'))


if __name__ == "__main__":
    for i in range(10):
        ablation(model_name='m5', test_group_num=i, dataset="TESS", high_freq_order=True)

