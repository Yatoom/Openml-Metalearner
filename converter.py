from sklearn.preprocessing import StandardScaler
import numpy as np

class Converter:
    UNKNOWN_TOKEN = "<unkn>"

    @staticmethod
    def to_float(x):
        return float(x)

    @staticmethod
    def array_to_numeric(x):
        if x == Converter.UNKNOWN_TOKEN:
            return []
        return [float(part) for part in x]

    @staticmethod
    def to_logical(x):
        return {"TRUE": 1, "FALSE": 0}.get(x.upper(), -1)

    @staticmethod
    def create_normalizer(distinct_values):
        scaler = StandardScaler()
        formatted = np.array(Converter.array_to_numeric(distinct_values)).reshape(-1, 1)
        scaler.fit(formatted)

        def transform(x):
            return scaler.transform(np.array(float(x)).reshape(-1, 1))[0][0]

        def inverse_transform(x):
            return int(scaler.inverse_transform(np.array(float(x)).reshape(-1, 1))[0][0])

        return transform, inverse_transform

    @staticmethod
    def create_category_mapper(distinct_values):
        key_values = zip(distinct_values, range(len(distinct_values)))
        dictionary = dict(key_values)

        def category_to_number(x):
            return dictionary.get(x, -1)

        return category_to_number
