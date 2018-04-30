import json
import numpy as np
import requests
import pandas as pd
from sklearn.preprocessing import StandardScaler

from better_converter import Converter

# Open files
params_json = open("flow_6794_100_params.json")
scores_json = open("flow_6794_100_scores.json")
quantities_json = open("openml_100_quantities.json")

# Load json
params_data = json.load(params_json)
scores_data = json.load(scores_json)
quantities_data = json.load(quantities_json)

# Handle parameters
description_data = requests.get("http://openml.org/api/v1/json/flow/6794").json()['flow']['parameter']
converter = Converter(params_data, description_data)
vectors = np.array(converter.get_vectors())

# Handle scores
scores = np.array(list(scores_data.values())).flatten()

# Handle quantities
quantities_data = dict([(i, quantities_data[i]) for i in params_data.keys()])
quantities_frame = pd.DataFrame(list(quantities_data.values())).dropna(axis=1)
quantities_frame = quantities_frame.apply(lambda x: StandardScaler().fit_transform(np.array(x).reshape(-1, 1)).flatten())
quantities = np.array(quantities_frame)

# Data
X = zip(vectors, quantities.repeat(100))
y = scores

