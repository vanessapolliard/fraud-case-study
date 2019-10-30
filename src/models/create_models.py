import os
import sys
sys.path.append('.')

import click
import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from src.data.make_dataset import load_data_as_dataframe

FILE_DIRECTORY = os.path.split(os.path.realpath(__file__))[0]  # Directory this script is in
SRC_DIRECTORY = os.path.split(FILE_DIRECTORY)[0]  # The 'src' directory
ROOT_DIRECTORY = os.path.split(SRC_DIRECTORY)[0]  # The root directory for the project
DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'data')  # The data directory
DATA_DIRECTORY_RAW = os.path.join(DATA_DIRECTORY, 'raw')  # The data/raw directory
DATA_DIRECTORY_PROCESSED = os.path.join(DATA_DIRECTORY, 'processed')  # The data/processed directory


# Load data
df_train = load_data_as_dataframe(name='train')

# Pull out X and y
target_name = 'acct_type'
y = df_train[target_name].str.contains('fraud').values.astype(int)
X = df_train.drop(target_name, axis=1)
feature_names = X.columns

# Create preprocessing transformer
ct = ColumnTransformer(
    transformers=[
        ('features', StandardScaler(), ['body_length', 'channels'])
    ])

transformed = ct.fit_transform(X)
