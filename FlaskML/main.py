from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.cluster import KMeans
import json

app = Flask(__name__)

A_clf_path = 'models/A_model.pkl'
with open(A_clf_path, 'rb') as f:
    A_clf = pickle.load(f)

B_clf_path = 'models/B_model.pkl'
with open(B_clf_path, 'rb') as f:
    B_clf = pickle.load(f)

C_clf_path = 'models/C_model.pkl'
with open(C_clf_path, 'rb') as f:
    C_clf = pickle.load(f)

@app.route('/')
def home():
    return "service up"

@app.route('/predict', methods=["POST"])
def predict():
    try:
        request_data = request.get_json()

        i = request_data["answers"]
        monitoring = sum(i[0:2])
        knowledge = sum(i[2:5])
        A = np.array([monitoring, knowledge]).reshape(1, -1)

        reaction_to_loss = i[5]+i[8]
        returns_expectation = i[5]+i[6]
        B = np.array([reaction_to_loss,returns_expectation]).reshape(1, -1)
        
        liquidity = i[7]+i[5]
        long_term = i[9]+i[5]
        C = np.array([liquidity, long_term]).reshape(1, -1)

        A_ypred = A_clf.predict(A)
        B_ypred = B_clf.predict(B)
        C_ypred = C_clf.predict(C)

        with open('profiles.json') as json_file:
            data = json.load(json_file)

            A_profile = data["A"][A_ypred[0]]
            B_profile = data["B"][B_ypred[0]]
            C_profile = data["C"][C_ypred[0]]

            result = {
                "title": A_profile[0] + ", " B_profile[0] + " e " + C_profile[0],
                "description": A_profile[1] + "\n\n" + B_profile[1] + "\n\n" + C_profile[1]
                }
    
        return jsonify(result)
    except:
        return "wrong format sent"