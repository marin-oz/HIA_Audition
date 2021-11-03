from data_processing.make_filtered_datasets import make_filtered
from data_processing.process_data import transform_data

# List of frequencies for low-pass filtered dataset
LIST_FREQUENCIES = [500, 550, 600, 650, 700, 750, 1000, 1250, 1500, 1750, 2000]
datasets =['TESS']

# Generate low-pass filtered datasets
for dataset in datasets:
    transform_data(dataset, 0)
    for freq in LIST_FREQUENCIES:
        make_filtered(dataset, freq)
        transform_data(dataset, freq)



