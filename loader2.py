import json

import requests
from tqdm import tqdm
import numpy as np
import os


class Loader:
    OPENML_100 = [3510, 36, 24, 9970, 9981, 2075, 34536, 34539, 21, 14964, 20, 14966, 14968, 14967, 146607, 125923,
                  34538, 9976, 14971, 9954, 9964, 3022, 125920, 14969, 125922, 3492, 23, 29, 28, 41, 37, 43, 22, 45, 58,
                  32, 9950, 49, 2074, 3560, 125921, 3903, 3902, 34537, 9952, 2079, 53, 9955, 9956, 9960, 3021, 9985,
                  3485, 10093, 10101, 3494, 3493, 14965, 3549, 18, 15, 16, 3567, 3561, 3543, 14, 12, 11, 3512, 3889,
                  9983, 2, 9980, 219, 14970, 3481, 7592, 146195, 9977, 3896, 3891, 3899, 3904, 3946, 3918, 3913, 3948,
                  3917, 3954, 3, 9914, 9946, 9957, 9967, 9968, 9978, 31, 6, 9979, 9971, 9986]

    BASE = "http://openml.org/api/v1/json"

    @staticmethod
    def load_task_qualities():
        """
        Loads the qualities for all tasks in the OPENML_100 set into a JSON file.
        :return: None
        """

        if os.path.isfile('data/openml_100_quantities.json'):
            print("File already exists")
            return

        q = Loader._load_all_qualities()
        with open('data/openml_100_quantities.json', 'w') as outfile:
            json.dump(q, outfile)

    @staticmethod
    def load_flow_data(flow_id):
        """
        Load parameters and scores data for a flow into JSON files.
        :param flow_id: (int) flow id
        :return: None
        """

        all_params, all_scores = Loader._load_flow_runs(flow_id)
        with open(f'data/flow_{flow_id}_params.json', 'w') as outfile:
            json.dump(all_params, outfile)

        with open(f'data/flow_{flow_id}_scores.json', 'w') as outfile:
            json.dump(all_scores, outfile)

    @staticmethod
    def _load_all_qualities():
        """
        Loads the qualities for all tasks in the OPENML_100 set.
        :return: (dict) a mapping of tasks to their qualities
        """
        qualities = {}
        for i in tqdm(Loader.OPENML_100):
            qualities[i] = Loader._load_qualities_for_task(i)
        return qualities

    @staticmethod
    def _load_qualities_for_task(task_id):
        """
        Load qualities of the dataset that belongs to the task.
        :param task_id: (int) task id
        :return: (dict) a mapping of quality names to their values
        """
        task_data = requests.get(
            f"{Loader.BASE}/task/{task_id}"
        ).json()

        dataset_id = task_data['task']['input'][0]['data_set']['data_set_id']

        dataset_data = requests.get(
            f"{Loader.BASE}/data/qualities/{dataset_id}"
        ).json()

        data = dataset_data['data_qualities']['quality']
        filtered = [i for i in data if not (isinstance(i['value'], list) or np.isnan(float(i['value'])))]

        qualities = dict([(i['name'], float(i['value'])) for i in filtered])

        return qualities

    @staticmethod
    def _load_flow_runs(flow_id, max_per_task=100):
        """
        Load evaluations of a flow for a set of tasks (OpenML 100).
        :param flow_id: (int) flow id
        :param max_per_task: (int) maximum number of runs to load per task
        :return: (dict) params and (dict) scores
        """
        params = {}
        scores = {}
        for i in tqdm(range(len(Loader.OPENML_100))):
            task_id = Loader.OPENML_100[i]
            task_params, task_scores = Loader._get_evaluations(task_id, flow_id, max_per_task)

            if len(task_params) > 0:
                params[task_id] = task_params
                scores[task_id] = task_scores

        return params, scores

    @staticmethod
    def _get_evaluations(task_id, flow_id, maximum):
        """
        Load random evaluations of a flow on a task.
        :param task_id: (int) task id
        :param flow_id: (int) flow id
        :param maximum: (int) maximum number of evaluations to load
        :return: (list) a list of parameter settings and (list) a list of scores
        """
        r = requests.get(
            f"{Loader.BASE}/evaluation/list/task/{task_id}/function/area_under_roc_curve/flow/{flow_id}/limit/10000"
        ).json()

        if "error" in r:
            return [], []

        evaluations = r['evaluations']['evaluation']

        chosen = np.random.choice(evaluations, maximum, replace=False)

        values = [Loader._get_param_values(i['setup_id']) for i in chosen]
        scores = [i['value'] for i in chosen]
        return values, scores

    @staticmethod
    def _get_param_values(setup_id):
        """
        Load the parameter values for a setup.
        :param setup_id: (int) setup id
        :return: (dict) a mapping of parameter names to values
        """
        return dict([
            (i['parameter_name'], i['value']) for i in
            requests.get(f"{Loader.BASE}/setup/{setup_id}").json()['setup_parameters']['parameter']
        ])
