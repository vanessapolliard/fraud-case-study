import os
import sys
sys.path.append('.')

import pickle

from sklearn.metrics import roc_curve, auc
from src.data.make_dataset import load_data_as_dataframe
from src.models.save_model_info import save_model_info
from src.features.featurize_data import featurize_data

import matplotlib.pyplot as plt


FILE_DIRECTORY = os.path.split(os.path.realpath(__file__))[0]  # Directory this script is in
SRC_DIRECTORY = os.path.split(FILE_DIRECTORY)[0]  # The 'src' directory
ROOT_DIRECTORY = os.path.split(SRC_DIRECTORY)[0]  # The root directory for the project
MODELS_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'models')  # Directory for pickled models and model info
DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'data')  # The data directory
DATA_DIRECTORY_RAW = os.path.join(DATA_DIRECTORY, 'raw')  # The data/raw directory
DATA_DIRECTORY_PROCESSED = os.path.join(DATA_DIRECTORY, 'processed')  # The data/processed directory

MODELS_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'models')  # Directory for pickled models and model info
model_filepath = os.path.join(MODELS_DIRECTORY, '8920304173528512454.pkl')


if __name__ == '__main__':
    df_test = load_data_as_dataframe(name='test')
    target_name = 'fraud'
    y = df_test[target_name]
    X = df_test.drop(target_name, axis=1)
    X = featurize_data(X)



    with open(model_filepath, 'rb') as f:
        model = pickle.load(f)


    #unique_model_id = abs(hash(str(model)))
    # unique_model_id = 'TEST'
    # save_model_info(model=model, unique_id=unique_model_id, X=X, y=y)