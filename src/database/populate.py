import os
import sys
sys.path.append('.')

import pickle
import time

import numpy as np
import pandas as pd
import psycopg2
import requests

from src.features.featurize_data import featurize_data

FILE_DIRECTORY = os.path.split(os.path.realpath(__file__))[0]  # Directory this script is in
SRC_DIRECTORY = os.path.split(FILE_DIRECTORY)[0]  # The 'src' directory
ROOT_DIRECTORY = os.path.split(SRC_DIRECTORY)[0]  # The root directory for the project
MODELS_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'models')  # Directory for pickled models and model info
DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'data')  # The data directory
DATA_DIRECTORY_RAW = os.path.join(DATA_DIRECTORY, 'raw')  # The data/raw directory
DATA_DIRECTORY_PROCESSED = os.path.join(DATA_DIRECTORY, 'processed')  # The data/processed directory

api_url = 'http://galvanize-case-study-on-fraud.herokuapp.com/data_point'

# Load Model
model_filepath = os.path.join(MODELS_DIRECTORY, '8920304173528512454.pkl')
with open(model_filepath, 'rb') as f:
    model = pickle.load(f)

# Set up database connection
conn = psycopg2.connect(database="fraud",
                        user="postgres",
                        host="localhost", port="5435")
cur = conn.cursor()

def insert_into_db(event_id, fraud, event_name):
    insert_vals = [event_id, fraud, event_name]
    insert_query = "INSERT INTO fraudstream VALUES \
                    (%s, %s, %s)"

    cur.execute(insert_query, tuple(insert_vals))
    conn.commit()

def does_event_id_exist(event_id):
    vals = [event_id]
    select_query = "SELECT COUNT(*) FROM fraudstream WHERE id = %s"

    cur.execute(select_query, tuple(vals))
    id_count = cur.fetchall()[0][0]

    if id_count == 0:
        return False
    else:
        return True

def single_api_query(link):
    response = requests.get(link)
    if response.status_code != 200:
        print('WARNING', response.status_code)
        return None
    else:
        api_response = response.json()
        return api_response


if __name__ == "__main__":
    while True:
        r = single_api_query(api_url)
        event_id = r['object_id']
        event_name = r['name']
        if does_event_id_exist(event_id):
            print(f'{event_id} already exists in database')
        else:
            df_sample = pd.DataFrame([r])
            df_sample = featurize_data(df_sample)
            try:
                predicted_proba = model.predict_proba(df_sample)[0][-1]
            except Exception as e:
                print(e)
                try:
                    df_sample = df_sample.fillna(np.nan)
                    predicted_proba = model.predict_proba(df_sample)[0][-1]
                except Exception as e:
                    print(e)
                    print('failed twice, passing this data point')
                    continue
            print(f'inserting event {event_id} into db with predicted fraud probability {predicted_proba:.3f}')
            insert_into_db(event_id, predicted_proba, event_name)

        time.sleep(3)

    # Close database connection
    conn.close()