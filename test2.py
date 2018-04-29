import json
import pandas as pd
from better_converter import Converter
import requests

with open('flow_6794_100_random.json', 'r') as infile:
    params_data = json.load(infile)

description_data = requests.get("http://openml.org/api/v1/json/flow/6794").json()['flow']['parameter']
conv = Converter(params_data, description_data)
result = conv.transform()
inverse = conv.inverse(result)
flats = conv.get_vectors(result)
sizes = conv.get_item_sizes()
reconstruction = conv.reconstruct(sizes, flats[0])

print("data")