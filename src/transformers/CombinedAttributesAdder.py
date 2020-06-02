from sklearn.base import BaseEstimator, TransformerMixin


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        X["rooms_per_household"] =  X["total_rooms"] / X["households"]
        X["population_per_household"]  = X["population"] / X["households"]
        if self.add_bedrooms_per_room:
            X["add_bedrooms_per_room"] = X["total_bedrooms"] / X["total_rooms"]
        return X

    def get_new_attr_created(self):
        if self.add_bedrooms_per_room:
            return [ "rooms_per_household", "population_per_household",
                                     "bedrooms_per_room"]
        else: return  ["rooms_per_household", "population_per_household"]
