import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize the Kaggle API
api = KaggleApi()
api.authenticate()

# Define the dataset URL or identifier
dataset_url = 'your-dataset-url-here'

# Extract the dataset identifier from the URL
dataset_identifier = dataset_url.split('/')[-1]

# Download the dataset
api.dataset_download_files(dataset_identifier, path='path/to/download', unzip=True)

print("Dataset downloaded successfully!")
