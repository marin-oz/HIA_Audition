import pathlib
import sys
import os
import neptune.new as neptune
import pickle

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from neptune.new.integrations.tensorflow_keras import NeptuneCallback

from models.load_data import DataLoader
from models.conv_models import ModelConstruction
from definition import DatasetDefinition

sys.path.append('..')


def train(model_name='m5', start_file="", filter=0, num_epochs=10, test_group_num=0, dataset="TESS"):

    run = neptune.init(project=os.environ['GITHUB_USER_NAME']+'/HIAAudition',
                       source_files=['*.py'])
   
    if filter == 0:
        save_name = 'original_'
    else:
        save_name = 'lowpass'+str(filter)+'Hz_'

    PARAMS = {'batch_size' : 32,
              'lr' : 0.01,
              'num_epochs' : num_epochs,
              'lowpass filter': filter,
              'model_name': model_name,
              'test_group_num': test_group_num,
              'dataset': dataset,
              'start_file': start_file,
              'save_name': save_name}
    
    run["sys/tags"].add([dataset, 'satori'])
    run["params"] = PARAMS

    data_def = DatasetDefinition(dataset, filter)
    data_def.set_test_group(test_group_num)

    model = ModelConstruction(model_name, data_def).model(kernel_size = data_def.kernel_size)
    
    respath = os.path.join(data_def.result_model_path, model_name, str(data_def.test_group_num))

    p = pathlib.Path(respath)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    
    if len(start_file)>0:
        model.load_weights(os.path.join(respath, start_file))
        save_name = os.path.join(respath, start_file + '_' + save_name)
    else:
        save_name = os.path.join(respath, save_name)
    
    model.compile(optimizer=SGD(learning_rate=PARAMS['lr']),loss='categorical_crossentropy',metrics=['acc'])
    print(model.summary())
    
    model_checkpoint_callback = ModelCheckpoint(
        filepath=save_name+'{epoch:03d}',save_weights_only=True,save_best_only=False,save_freq='epoch',period=10)
    
    neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')
    
    batch_size = PARAMS['batch_size']

    data_loader = DataLoader(data_def)
    x_train, y_train = data_loader.get_training_data()
    x_test, y_test = data_loader.get_testing_data()
    print('x_train.shape =', x_train.shape,flush=True)
    print('y_train.shape =', y_train.shape,flush=True)
    print('x_test.shape =', x_test.shape,flush=True)
    print('y_test.shape =', y_test.shape,flush=True)

    model.fit(x=x_train,y=y_train,batch_size=batch_size,epochs=num_epochs,verbose=2,shuffle=True,
              callbacks=[neptune_cbk,model_checkpoint_callback],validation_data=(x_test, y_test))

    pickle.dump(PARAMS, open(save_name+"params.pickle", "wb"))


if __name__ == '__main__':
    train()