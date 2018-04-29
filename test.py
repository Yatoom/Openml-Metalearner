import json

import numpy as np
import pandas as pd
import requests

# key_superset = set([keys for task in params.values() for evaluation in task for keys in evaluation.keys()])
# description = requests.get("http://openml.org/api/v1/json/flow/6794").json()['flow']['parameter']
# key_superset = [i['name'] for i in description]
# types = [i['data_type'] for i in description]
# print(params)
from converter import Converter

ranger_defaults = {
    "num.trees": 500,
    "replace": "TRUE",
    "respect.unordered.factors": "ignore",
    "importance": "none",
    "write.forest": "TRUE",
    "scale.permutation.importance": "FALSE",
    "save.memory": "FALSE",
    "verbose": "TRUE",
    "splitrule": "gini",
    "num.random.splits": 1,
    "keep.inbag": "FALSE"
}


def superset(params, param_description):
    key_superset = [i['name'] for i in param_description]
    # key_superset2 = set([k for i in params.values() for j in i for k in j.keys()])

    # Nan defaults
    defaults_categorical = dict([(i['name'], Converter.UNKNOWN_TOKEN) for i in param_description])
    defaults_negative = dict([(i['name'], -1) for i in param_description if i['data_type'] in ["numeric", "integer"]])
    defaults_unknown = dict(defaults_categorical, **defaults_negative)

    # Openml defaults
    defaults_openml = dict([(i['name'], i['default_value']) for i in param_description if i['default_value']])

    # Model defaults
    defaults_model = dict([(i, j) for i, j in ranger_defaults.items() if i in key_superset])

    # Final defaults
    defaults = dict(dict(defaults_unknown, **defaults_openml), **defaults_model)

    expanded_result = {}
    for task_id, runs in params.items():
        expanded_runs = []
        for evaluation in runs:
            expanded_evaluation = dict(defaults, **evaluation)
            expanded_runs.append(expanded_evaluation)
        expanded_result[task_id] = expanded_runs

    return expanded_result


def get_distinct_values(params, param, numeric):
    all_settings = [i for task_id, evaluations in params.items() for i in evaluations]
    distinct = pd.DataFrame(all_settings)[param].unique()
    if numeric:
        distinct = pd.to_numeric(distinct)

    return distinct.tolist()


def create_mapping(params, description):
    mapping = {}
    for i in description:
        data_type = i['data_type']
        name = i['name']
        if data_type == "discrete" or data_type == "untyped":
            distinct = get_distinct_values(params, i['name'], False)
            key_values = zip(distinct, range(len(distinct)))
            dictionary = dict(key_values)
            mapping[name] = lambda x: dictionary.get(x, -1)
        elif data_type == "logical":
            mapping[name] = lambda x: {"TRUE": 1, "FALSE": 0}.get(x, -1)
        elif data_type == "integer":
            distinct = get_distinct_values(params, i['name'], True)
            mapping[name] = Converter.create_normalizer(distinct)
        elif data_type == "numeric":
            mapping[name] = Converter.to_float
        elif data_type == "numericvector":
            mapping[name] = Converter.array_to_numeric

    return mapping


def map_params(params, mapping):
    mapped_result = {}
    for task_id, runs in params.items():
        mapped_runs = []
        for evaluation in runs:
            mapped_key_values = [(i, mapping[i](j)) for i, j in evaluation.items()]
            mapped_evaluation = dict(mapped_key_values)
            mapped_runs.append(mapped_evaluation)
        mapped_result[task_id] = mapped_runs

    return mapped_result


def preprocess_params(params, description):
    # Get distinct values

    # Logical to 0/1
    pass

    # Discrete to one hot encoded
    pass

    # Integer to normalized numeric
    pass

    # Numeric
    pass


def flatten_params(params):
    return [i for task_id, evaluations in params.items() for i in evaluations]


with open('flow_6794_100_random.json', 'r') as infile:
    params_data = json.load(infile)

description_data = requests.get("http://openml.org/api/v1/json/flow/6794").json()['flow']['parameter']

s_params = superset(params_data, description_data)
map = create_mapping(s_params, description_data)
mapped = map_params(s_params, map)
p_params = preprocess_params(s_params, description_data)
print(map)
