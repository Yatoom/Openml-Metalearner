import json
import pandas as pd
from better_converter import Converter
import requests

with open('flow_6794_100_random.json', 'r') as infile:
    params_data = json.load(infile)

description_data = requests.get("http://openml.org/api/v1/json/flow/6794").json()['flow']['parameter']
conv = Converter(params_data, description_data)
vectors = conv.get_vectors()
params = conv.get_params(vectors)
print("data")