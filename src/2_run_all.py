from definition import DatasetDefinition
from models.train_model import train
from models.test_model import model_test_all
from models.test_ablation import ablation

dataset = "TESS"
model = 'm5'

data_def = DatasetDefinition(dataset,0)
num_fold = data_def.n_data_split_groups

# Train the model for 10 folds
for i in range(num_fold):
    train(model_name=model, start_file="", filter=0, num_epochs=100, test_group_num=i, dataset=dataset)
    train(model_name=model, start_file="", filter=500, num_epochs=100, test_group_num=i, dataset=dataset)
    train(model_name=model, start_file="lowpass500Hz_050", filter=0, num_epochs=50, test_group_num=i, dataset=dataset)
    train(model_name=model, start_file="original_050", filter=500, num_epochs=50, test_group_num=i, dataset=dataset)

# Test the model for 10 folds
for i in range(num_fold):
    model_test_all(model_name=model, test_group_num=i, dataset=dataset)
    ablation(model_name=model, test_group_num=i, dataset=dataset, high_freq_order=True, random_order=False)

