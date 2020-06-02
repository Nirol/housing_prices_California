from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler

CAPPED_VALUE_DICT = {
    'median_house_value':  500001.0,
    'housing_median_age':  52.0

}

class CappedValuesRemover:
    def  __init__(self, feature_to_remove):
        self.feature_to_remove = feature_to_remove
    def transform(self, housing):
        df_to_remove = pd.DataFrame()
        to_remove_samples = housing[housing[self.feature_to_remove] == CAPPED_VALUE_DICT[self.feature_to_remove]]
        df_to_remove = df_to_remove.append(to_remove_samples)
        housing_clean = pd.concat([housing, df_to_remove, df_to_remove]).drop_duplicates(
                        keep=False)
        housing_clean.reset_index(drop=True, inplace=True)
        return housing_clean


class AttributesRemover(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_remove):
        self.features_to_remove = features_to_remove
    def fit(self, X):
        return self
    def transform(self, X):
        X = np.delete(X, self.features_to_remove, 1)
        return X


class FeatureScaleTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformer:str =None ):
        self.transformer = transformer
    def fit(self, X):
        return self
    def transform(self, X):
        if self.transformer:
            if  self.transformer == "log":
                pow_trans  = PowerTransformer()
            elif self.transformer == "scale":
                pow_trans = StandardScaler()
            pow_trans.fit(X)
            trans_data  = pow_trans.transform(X)
            return trans_data
