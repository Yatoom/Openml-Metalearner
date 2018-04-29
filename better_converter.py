import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np


class Converter:

    def __init__(self, params, description):
        self.params = params
        self.description = dict([(i['name'], i) for i in description])
        self.frame = pd.DataFrame([j for i in params.values() for j in i])
        self.preprocessed = self._preprocess(self.frame)
        self.distinct = self._get_distinct_values(self.preprocessed)
        self.mapping = self._create_mapping(self.distinct)

    def _preprocess(self, frame):

        # Fill with default (to point out that missing parameters are set to its default)
        f = frame.copy()
        f = f.fillna("default")

        for i in self.frame.columns:
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
        for i in frame.columns:
            result[i] = sorted(frame[i].unique().tolist())
        return result

    def _create_transformer(self, transformer, reshape=False):
        def transform(x):
            r = x
            r = np.array(r).reshape(-1, 1) if reshape else r
            return transformer.transform(r)

        return transform

    def _create_encoder(self, transformer):
        def transform(x):
            length = len(transformer.classes_)
            return [self.one_hot(i, length) for i in transformer.transform(x)]

        return transform

    def _create_mapping(self, distinct):
        mapping = {}

        # Create mapping forward and back
        for i in self.frame.columns:
            data_type = self.description[i]['data_type']

            if data_type in ["numeric", "integer"]:
                transformer = self._create_transformer(StandardScaler().fit(np.array(distinct[i]).reshape(-1, 1)), reshape=True)
                mapping[i] = transformer

            if data_type in ["discrete"]:
                mapping[i] = self._create_encoder(LabelEncoder().fit(distinct[i]))

            if data_type in ["logical"]:
                mapping[i] = lambda x: [{"FALSE": 0, "TRUE": 1}.get(i, 0.5) for i in x]

        return mapping

    @staticmethod
    def one_hot(hot_index, length):
        return [0 if i != hot_index else 1 for i in range(length)]

    def transform(self):
        f = self.preprocessed.copy()
        for i in f.columns:
            f[i] = self.mapping[i](f[i])
        return f

    def get_flat(self, frame):
        values = [i.values() for i in frame.to_dict(orient="records")]

        result = []
        for i in values:
            result.append(self._flatten_iterables(i))

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
