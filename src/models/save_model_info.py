import os
import sys
sys.path.append('.')

import click
import numpy as np
import pandas as pd
import pickle

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

FILE_DIRECTORY = os.path.split(os.path.realpath(__file__))[0]  # Directory this script is in
SRC_DIRECTORY = os.path.split(FILE_DIRECTORY)[0]  # The 'src' directory
ROOT_DIRECTORY = os.path.split(SRC_DIRECTORY)[0]  # The root directory for the project
MODELS_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'models')  # Directory for pickled models and model info
DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'data')  # The data directory
DATA_DIRECTORY_RAW = os.path.join(DATA_DIRECTORY, 'raw')  # The data/raw directory
DATA_DIRECTORY_PROCESSED = os.path.join(DATA_DIRECTORY, 'processed')  # The data/processed directory


def save_model_info(model, unique_id, X, y):
    # Save model information to our model info dataframe
    user = os.environ['USER']
    model_info_filepath = os.path.join(MODELS_DIRECTORY, f'model-info-{user}.csv')
    if os.path.exists(model_info_filepath):
        model_info = pd.read_csv(model_info_filepath)
    else:
        print('Need to create base model info dataframe')
        column_names = [
            "unique_id",
            "pickle_filename",
            "pipeline_string",
            "named_steps",
            "classification_threshold",
            "cv_accuracy",
            "cv_precision",
            "cv_recall",
            "cv_roc_auc",
            "cv_f1"
        ]
        model_info = pd.DataFrame(columns=column_names)

    pickle_filename = os.path.join(MODELS_DIRECTORY, f'{unique_id}.pkl')
    pipeline_string = str(model)
    named_steps = model.named_steps
    classification_threshold = 0.5
    cv_accuracy = np.mean(cross_val_score(model, X, y, cv=5))
    cv_precision = np.mean(cross_val_score(model, X, y, cv=5, scoring='precision'))
    cv_recall = np.mean(cross_val_score(model, X, y, cv=5, scoring='recall'))
    cv_roc_auc = np.mean(cross_val_score(model, X, y, cv=5, scoring='roc_auc'))
    cv_f1 = np.mean(cross_val_score(model, X, y, cv=5, scoring='f1'))

    new_row = {
        "unique_id": unique_id,
        "pickle_filename": pickle_filename,
        "pipeline_string": pipeline_string,
        "named_steps": named_steps,
        "classification_threshold": classification_threshold,
        "cv_accuracy": cv_accuracy,
        "cv_precision": cv_precision,
        "cv_recall": cv_recall,
        "cv_roc_auc": cv_roc_auc,
        "cv_f1": cv_f1
    }

    model_info = model_info.append(new_row, ignore_index=True)
    model_info.to_csv(model_info_filepath, index=False)