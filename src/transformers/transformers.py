from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


CAPPED_VALUE_DICT = {
    'median_house_value':  500001.0,
    'housing_median_age':  52.0

}


class CappedValuesRemover:
    def  __init__(self, feature_to_remove):  # no *args or **kargs
        self.feature_to_remove = feature_to_remove
    def transform(self, housing):
        df_to_remove = pd.DataFrame()
        to_remove_samples = housing[housing[self.feature_to_remove] == CAPPED_VALUE_DICT[self.feature_to_remove]]
        df_to_remove = df_to_remove.append(to_remove_samples)
        housing_clean = pd.concat([housing, df_to_remove, df_to_remove]).drop_duplicates(
                        keep=False)
        housing_clean.reset_index(drop=True, inplace=True)
        return housing_clean







CLEAN_OUTLIERS_DICT = {


    'hard_filter': {'features': ["total_rooms", "total_bedrooms"],
                    'upper_qurant_percent': 0.975, 'lower_qurant_percent': 0},
    'med_filter': {'features': ["households"], 'upper_qurant_percent': 0.9733,
                   'lower_qurant_percent': 0},
    'easy_filter': {'features': ["population"], 'upper_qurant_percent': 0.969,
                    'lower_qurant_percent': 0}}



def _samples_to_remove(isUpperValue, housing,feature, inner_dict):
    series_ = housing[feature]
    if isUpperValue:
        qurantile_value = series_.quantile(inner_dict['upper_qurant_percent'])
        return housing[housing[feature] > qurantile_value]
    else:
        qurantile_value = series_.quantile(inner_dict['lower_qurant_percent'])
        return housing[housing[feature] < qurantile_value]


class OutlinerRemover():
    def transform(self, housing, y=None):
        df_empty = pd.DataFrame()
        for inner_dict in CLEAN_OUTLIERS_DICT.values():
            for feature in inner_dict['features']:
                low_filtered_samples =_samples_to_remove(False, housing, feature, inner_dict)
                high_filtered_samples = _samples_to_remove(True, housing,
                                                          feature, inner_dict)
                df_empty = df_empty.append(low_filtered_samples)
                df_empty = df_empty.append(high_filtered_samples)
        housing_clean = pd.concat([housing, df_empty, df_empty]).drop_duplicates(
                        keep=False)
        return housing_clean



class AttributesRemover(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_remove): # no *args or **kargs
        self.features_to_remove = features_to_remove
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        X = np.delete(X, self.features_to_remove, 1)
        return X












rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


    def get_new_attr_created(self):
        if self.add_bedrooms_per_room:
            return [ "rooms_per_household", "population_per_household",
                                     "bedrooms_per_room"]
        else: return  ["rooms_per_household", "population_per_household"]

