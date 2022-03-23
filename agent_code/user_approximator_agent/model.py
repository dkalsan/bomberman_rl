import numpy as np
import pandas as pd

from typing import Any, Dict, List

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class RFAgent():
    def __init__(
        self,
        categorical_features: Dict[str, List[Any]],
        actions: List[str],
        config: Dict[str, Any],
    ):
        """
        We need to know which values categorical features can take in advance.
        Numerical features can be added at will during predict call and will be
        passed through.

        Example on how to structure categorical_features:
        {
            "cat_feature1": ["possible_value_1", ..., "possible_value_N"],
            ...,
            "cat_featureN": ["possible_value_1", ..., "possible_value_N"]
        }
        """

        if "n_jobs" in config.keys():
            n_jobs = config["n_jobs"]

        for k, v in categorical_features.items():
            assert len(v) > 2, f"Unsupported: Categorical feature '{k}' takes less than 2 possible values."

        self.model = Pipeline([
            ("categorical_transformer", ColumnTransformer(
                [(name, OneHotEncoder(categories=[values]), [name])
                 for name, values in categorical_features.items()],
                remainder="passthrough",
                n_jobs=n_jobs
            )),
            ("model", RandomForestRegressor(**config))
        ])

        self.actions = actions

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, dataframe: pd.DataFrame):
        try:
            return self.model.predict(dataframe)
        except NotFittedError:
            return np.zeros((1, len(self.actions)))

    def predict_action(self, dataframe: pd.DataFrame):
        return np.array(self.actions)[np.argmax(self.predict(dataframe), axis=1)]
