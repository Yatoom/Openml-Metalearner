from collections import OrderedDict

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np


class Converter:

    def __init__(self, params, description):
        self.params = params
        self.description = dict([(i['name'], i) for i in description])
        self.frame = pd.DataFrame([j for i in params.values() for j in i])
        self.columns = self.frame.columns.tolist()
        self.types = [self.description[i]['data_type'] for i in self.columns]
        self.preprocessed = self._preprocess(self.frame)
        self.distinct = self._get_distinct_values(self.preprocessed)
        self.num_distinct = OrderedDict(([(i, len(j)) for i, j in self.distinct.items()]))
        self.mapping, self.reverser = self._create_mapping(self.distinct)

    def _preprocess(self, frame):

        # Fill with default (to point out that missing parameters are set to its default)
        f = frame.copy()
        f = f.fillna("default")

        for i in self.columns:
            # Get data type
            data_type = self.description[i]['data_type']

            # Allowed types
            if data_type not in ["numeric", "integer", "discrete", "logical"]:
                continue

            if data_type in ["numeric"]:
                f[i] = [float(i) if i != "default" else -1 for i in f[i]]

            if data_type in ["integer"]:
                f[i] = [float(i) if i != "default" else -1 for i in f[i]]

        return f

    def _get_distinct_values(self, frame):
        result = {}
        for i in self.columns:
            result[i] = sorted(frame[i].unique().tolist())
        return result

    @staticmethod
    def _create_transformer(transformer):
        def transform(x):
            r = np.array(x).reshape(-1, 1)
            return transformer.transform(r)

        def inverse_transform(x):
            r = np.array(x).reshape(-1, 1)
            return transformer.inverse_transform(r)

        return transform, inverse_transform

    @staticmethod
    def _create_encoder(transformer):
        def transform(x):
            length = len(transformer.classes_)
            return [Converter.one_hot(i, length) for i in transformer.transform(x)]

        def inverse_transform(x):
            # Find the index of the 1.
            indices = [np.where(np.array(i) == 1)[0][0] for i in x]
            return transformer.inverse_transform(indices)

        return transform, inverse_transform

    def _create_mapping(self, distinct):
        mapping = {}
        reverse_mapping = {}

        # Create mapping forward and back
        for i in self.columns:
            data_type = self.description[i]['data_type']

            if data_type in ["numeric", "integer"]:
                transformer, reverser = self._create_transformer(StandardScaler().fit(np.array(distinct[i]).reshape(-1, 1)))
                mapping[i] = transformer
                reverse_mapping[i] = reverser

            if data_type in ["discrete"]:
                transformer, reverser = self._create_encoder(LabelEncoder().fit(distinct[i]))
                mapping[i] = transformer
                reverse_mapping[i] = reverser

            if data_type in ["logical"]:
                mapping[i] = lambda x: [{"FALSE": 0, "TRUE": 1}.get(i, 0.5) for i in x]
                reverse_mapping[i] = lambda x: ["FALSE" if i < 0.5 else "TRUE" if i > 0.5 else "default" for i in x]

        return mapping, reverse_mapping

    @staticmethod
    def one_hot(hot_index, length):
        return [0 if i != hot_index else 1 for i in range(length)]

    def transform(self):
        f = self.preprocessed.copy()
        for i in f.columns:
            f[i] = self.mapping[i](f[i])
        return f

    def inverse(self, frame):
        f = frame.copy()
        for i in self.columns:
            f[i] = self.reverser[i](f[i])
        return f

    def get_vectors(self, frame):
        values = [i.values() for i in frame.to_dict(orient="records")]

        result = []
        for i in values:
            result.append(self._flatten_iterables(i))

        return result

    def get_item_sizes(self):

        how_many = OrderedDict()
        for i in self.columns:
            how_many[i] = 1

            if self.description[i]['data_type'] in ["discrete"]:
                how_many[i] = self.num_distinct[i]

        return how_many

    def reconstruct(self, item_sizes, vector):
        vector = np.array(vector)
        result = {}
        i = 0
        for param, number in item_sizes.items():
            if number == 1:
                result[param] = vector[i]
            else:
                indices = range(i, i + number)
                result[param] = vector[indices]
            i += number
        return result


    @staticmethod
    def _flatten_iterables(values):
        """
        Makes sure all items are lists, and then flattens the list.
        :param values: A list of items, where each item can be a list or another item
        :return:
        """
        vectors = [[i] if not hasattr(i, "__iter__") else i for i in values]
        flat = [j for i in vectors for j in i]
        return flat
