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
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as IMPipeline

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

#####################################
# FEATURIZATION BLOCK
X['event_duration'] = X['event_end'] - X['event_start']
X['time_to_create'] = X['event_published'] - X['event_created']
X['org_desc_exists'] = np.where(X['org_desc'] != '', 1, 0)
X['org_name_exists'] = np.where(X['org_name'] != '', 1, 0)
X['num_previous_payouts'] = X['previous_payouts'].map(lambda x: len(x))
X['num_ticket_types'] = X['ticket_types'].map(lambda x: len(x))
X['venue_address_exists'] = np.where(X['venue_address'] != '', 1, 0)
X['venue_name_exists'] = np.where(X['venue_name'] != '', 1, 0)
X['email_in_top_five_domains'] = X['email_domain'].map(lambda x: x in ('gmail.com', 'yahoo.com', 'hotmail.com', 'aol.com', 'live.com'))
#####################################

# Create preprocessing transformer
scaled_features = ['body_length', 'channels', 'name_length', 'org_facebook', 'org_twitter', 'user_age','event_duration','time_to_create','num_previous_payouts','num_ticket_types']
passthrough_features = ['fb_published', 'has_analytics', 'has_header', 'has_logo', 'show_map','org_desc_exists','org_name_exists','venue_address_exists','venue_name_exists']
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
model = RandomForestClassifier(n_estimators=100, random_state=42)

oversampler = RandomOverSampler(
                sampling_strategy=1,
                random_state=42)

pipeline = IMPipeline([('ct', ct), ('oversampler', oversampler), ('model', model)])
search = None

#####################################
# GRID SEARCH BLOCK
# param_grid = {
#     'oversampler__sampling_strategy': [0.93, 0.95, 0.97, 1]
# }
# search = GridSearchCV(pipeline, param_grid, iid=False, cv=5)
# search.fit(X, y)
# print("Best parameter (CV score=%0.3f):" % search.best_score_)
# print(search.best_params_)
#####################################

# Fit our model using the best parameters
if search:
    pipeline.set_params(**search.best_params_).fit(X, y)
else:
    pipeline.set_params().fit(X, y)

unique_model_id = abs(hash(str(pipeline)))
save_model_info(model=pipeline, unique_id=unique_model_id, X=X, y=y)

# Save pickled model
with open(f"models/{unique_model_id}.pkl","wb") as f:
    pickle.dump(pipeline, f)