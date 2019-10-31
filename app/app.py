import os
import pickle

from flask import Flask, render_template, jsonify, request, redirect, url_for
import psycopg2

# FILE_DIRECTORY = os.path.split(os.path.realpath(__file__))[0]
# DATA_DIRECTORY = os.path.join(FILE_DIRECTORY, 'data')

app = Flask(__name__)

conn = psycopg2.connect(database="fraud",
                        user="postgres",
                        host="localhost", port="5435")
cur = conn.cursor()

def get_db_data():
    if os.environ['USER'] == 'stevenrouk':
        select_query = "SELECT * FROM fraudstream"
    else:
        select_query = "SELECT id, fraud, event_name FROM fraudstream"
    cur.execute(select_query)
    data = cur.fetchall()
    if os.environ['USER'] == 'stevenrouk':
        data = sorted(data, key=lambda x: x[1], reverse=True)
    else:
        data = sorted(data, key=lambda x: x[2], reverse=True)

    return data

# with open(os.path.join(DATA_DIRECTORY, 'user_recommendations.pkl'), 'rb') as f:
#     user_recommendations = pickle.load(f)

# with open(os.path.join(DATA_DIRECTORY, 'user_fave_movies.pkl'), 'rb') as f:
#     user_favorites = pickle.load(f)

# home page
@app.route('/')
def index():
    data = get_db_data()
    return render_template('index.html', data=data)

if __name__ == '__main__':
    if os.environ['USER'] == 'stevenrouk':
        debug = True
    else:
        debug = False
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=True) # Make sure to change debug=False for production
