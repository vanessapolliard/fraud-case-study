import os
import pickle

from flask import Flask, render_template, jsonify, request, redirect, url_for

# FILE_DIRECTORY = os.path.split(os.path.realpath(__file__))[0]
# DATA_DIRECTORY = os.path.join(FILE_DIRECTORY, 'data')

app = Flask(__name__)

# with open(os.path.join(DATA_DIRECTORY, 'user_recommendations.pkl'), 'rb') as f:
#     user_recommendations = pickle.load(f)

# with open(os.path.join(DATA_DIRECTORY, 'user_fave_movies.pkl'), 'rb') as f:
#     user_favorites = pickle.load(f)

# home page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=False) # Make sure to change debug=False for production
