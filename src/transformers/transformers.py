from typing import List

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler

from src.project_settings import CAPPED_VALUE_DICT


class CappedValuesRemover:
    """
    Transformer to remove capped value froma a feature, based on project setting CAPPED_VALUE_DICT.
    Attributes
    ----------
    feature_to_remove: the feature we clean capped value districts from.

    Methods
    -------
    transform(housing):
        Transform the dataframe data, returning filtered dataframe.

    """
    def  __init__(self, feature_to_remove: str):
        self.feature_to_remove = feature_to_remove
    def transform(self, housing: pd.DataFrame)->  pd.DataFrame:
        df_to_remove = pd.DataFrame()
        to_remove_samples = housing[housing[self.feature_to_remove] == CAPPED_VALUE_DICT[self.feature_to_remove]]
        df_to_remove = df_to_remove.append(to_remove_samples)
        housing_clean = pd.concat([housing, df_to_remove, df_to_remove]).drop_duplicates(
                        keep=False)
        housing_clean.reset_index(drop=True, inplace=True)
        return housing_clean





class FeatureScaleTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to scale the dataframe features by either PowerTransformer (log
    transform) or StandardScaler (standardization).
    Attributes
    ----------
    transformer: either "log" or "scale" string to determine the type of feature
    scaling.

    Methods
    -------
    transform(housing):
        Transform the dataframe data, returning filtered dataframe.

    """
    def __init__(self, transformer:str =None ):
        self.transformer = transformer
    def fit(self):
        return self
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.transformer:
            if  self.transformer == "log":
                pow_trans  = PowerTransformer()
            elif self.transformer == "scale":
                pow_trans = StandardScaler()
            pow_trans.fit(X)
            trans_data  = pow_trans.transform(X)
            return trans_data
