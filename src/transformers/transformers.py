from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


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

