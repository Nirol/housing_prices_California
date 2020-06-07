from typing import List

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """
    Combine existing dataframe columns (Features) into new ones.
    Used as internal part of pipeline_attribute_adder pipeline.

    As of now the newly created features are hardcoded into the transformer class.

    Control over the newly created features are set over src/project_settings.py
    and are used over at the pipeline method.

    Attributes
    ----------
    add_bedrooms_per_room a binary flag attribute if the transformer will
    create bedrooms_per_room feature.

    Methods
    -------
    transform(X):
        Create the new features.

    get_new_attr_created():
        return ordered list of the newly created feature names, used to
        to name the added colu,ms to the dataframe.
    """
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self):
        return self # nothing else to do
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X["rooms_per_household"] =  X["total_rooms"] / X["households"]
        X["population_per_household"]  = X["population"] / X["households"]
        if self.add_bedrooms_per_room:
            X["bedrooms_per_room"] = X["total_bedrooms"] / X["total_rooms"]
        return X

    def get_new_attr_created(self)-> List[str]:
        if self.add_bedrooms_per_room:
            return [ "rooms_per_household", "population_per_household",
                                     "bedrooms_per_room"]
        else: return  ["rooms_per_household", "population_per_household"]
