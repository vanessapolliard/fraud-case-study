import os
import sys
sys.path.append('.')

import click
import numpy as np
import pandas as pd
import pickle

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

from src.data.make_dataset import load_data_as_dataframe
from src.models.save_model_info import save_model_info


FILE_DIRECTORY = os.path.split(os.path.realpath(__file__))[0]  # Directory this script is in
SRC_DIRECTORY = os.path.split(FILE_DIRECTORY)[0]  # The 'src' directory
ROOT_DIRECTORY = os.path.split(SRC_DIRECTORY)[0]  # The root directory for the project
MODELS_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'models')  # Directory for pickled models and model info
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
scaled_features = ['body_length', 'channels', 'name_length', 'org_facebook', 'org_twitter', 'user_age']
passthrough_features = ['fb_published', 'has_analytics', 'has_header', 'has_logo', 'show_map']
onehot_features = ['delivery_method', 'user_type']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=-999)),
    ('onehot', OneHotEncoder(handle_unknown='ignore', categories='auto'))])  # getting rid of drop='first' for now
passthrough_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=-999))])

ct = ColumnTransformer(
    transformers=[
        ('scaled_features', numeric_transformer, scaled_features),
        ('passthrough_features', passthrough_transformer, passthrough_features),
        ('onehot_features', categorical_transformer, onehot_features)
    ])

#model = LogisticRegression(solver='lbfgs')
model = RandomForestClassifier(n_estimators=100)

pipeline = Pipeline([('ct', ct), ('model', model)])
pipeline.set_params().fit(X, y)

print(f"basic score: {pipeline.score(X, y)}")
print(f"CV score: {cross_val_score(pipeline, X, y, cv=5)}")

unique_model_id = abs(hash(str(pipeline)))
# save_model_info(model=pipeline, unique_id=unique_model_id, X=X, y=y)

# Save pickled model
with open(f"models/{unique_model_id}.pkl","wb") as f:
    pickle.dump(pipeline, f)