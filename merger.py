import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from better_converter import Converter


class Merger:

    def __init__(self, params, description, scores, qualities):
        # Accept parameters
        self.params = params
        self.description = description
        self.scores = scores
        self.qualities = qualities

        # Get tasks
        self.tasks = self.params.keys()

        # Qualities to data frame
        qualities_data = dict([(i, self.qualities[i]) for i in self.tasks])
        self.qualities_frame = pd.DataFrame(list(qualities_data.values())).dropna(axis=1)

        # Create converter
        self.converter = Converter(params, description)

        # Separate converters for qualities
        self.scalers = dict([
            (i, StandardScaler().fit(np.array(self.qualities_frame[i]).reshape(-1, 1)))
            for i in self.qualities_frame.columns
        ])

    # Merge qualities and parameters together
    def merge(self, runs_per_task=100):
        vectors = np.array(self.converter.get_vectors())
        scores = np.array(list(self.scores.values())).flatten()

        qualities: pd.DataFrame = self.qualities_frame.copy()
        for i in qualities.columns:
            qualities[i] = self.scalers[i].transform(np.array(qualities[i]).reshape(-1, 1)).flatten()

        # Data
        X = np.concatenate((vectors, np.array(qualities).repeat(runs_per_task, axis=0)), axis=1)
        y = scores

        return X, y
