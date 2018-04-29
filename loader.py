import json

import numpy as np
import requests
from tqdm import tqdm

OPENML_100 = [3510, 36, 24, 9970, 9981, 2075, 34536, 34539, 21, 14964, 20, 14966, 14968, 14967, 146607, 125923, 34538,
              9976, 14971, 9954, 9964, 3022, 125920, 14969, 125922, 3492, 23, 29, 28, 41, 37, 43, 22, 45, 58, 32, 9950,
              49, 2074, 3560, 125921, 3903, 3902, 34537, 9952, 2079, 53, 9955, 9956, 9960, 3021, 9985, 3485, 10093,
              10101, 3494, 3493, 14965, 3549, 18, 15, 16, 3567, 3561, 3543, 14, 12, 11, 3512, 3889, 9983, 2, 9980, 219,
              14970, 3481, 7592, 146195, 9977, 3896, 3891, 3899, 3904, 3946, 3918, 3913, 3948, 3917, 3954, 3, 9914,
              9946, 9957, 9967, 9968, 9978, 31, 6, 9979, 9971, 9986]

BASE = "http://openml.org/api/v1/json"


def load_flow_runs(flow_id, max_per_task=100):
    params = {}
    scores = {}
    for i in tqdm(range(len(OPENML_100))):
        task_id = OPENML_100[i]
        task_params, task_scores = get_evaluations(task_id, flow_id, max_per_task)

        if len(task_params) > 0:
            params[task_id] = task_params
            scores[task_id] = task_scores

    return params, scores


def get_evaluations(task_id, flow_id, maximum):
    r = requests.get(
        f"{BASE}/evaluation/list/task/{task_id}/function/area_under_roc_curve/flow/{flow_id}/limit/10000"
    ).json()

    if "error" in r:
        return [], []

    evaluations = r['evaluations']['evaluation']

    chosen = np.random.choice(evaluations, maximum, replace=False)

    values = [get_param_values(i['setup_id']) for i in chosen]
    scores = [i['value'] for i in chosen]
    return values, scores


def get_param_values(setup_id):
    return dict([
        (i['parameter_name'], i['value']) for i in
        requests.get(f"http://openml.org/api/v1/json/setup/{setup_id}").json()['setup_parameters']['parameter']
    ])


all_params, all_scores = load_flow_runs(6794)
with open('flow_6794_100_params.json', 'w') as outfile:
    json.dump(all_params, outfile)

with open('flow_6794_100_scores.json', 'w') as outfile:
    json.dump(all_scores, outfile)
