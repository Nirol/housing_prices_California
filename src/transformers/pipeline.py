from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelBinarizer

from my_proj.src.transformers.combined_attr_transformer import \
    CombinedAttributesAdder

from sklearn.pipeline import FeatureUnion
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
 def __init__(self, attribute_names):
    self.attribute_names = attribute_names
 def fit(self, X, y=None):
    return self
 def transform(self, X):
     return X[self.attribute_names].values


class SupervisionFriendlyLabelBinarizer(LabelBinarizer):
    def fit_transform(self, X, y=None):
        return super(SupervisionFriendlyLabelBinarizer,self).fit_transform(X)



def from_array_to_df(housing_prepared_array, num_attribs, combined_attr_adder, one_hot_encode_list  ):
    newly_created_attr = combined_attr_adder.get_new_attr_created()


    all_columns_list = num_attribs + newly_created_attr + one_hot_encode_list
    housing_prepared_df = pd.DataFrame(housing_prepared_array, columns=all_columns_list)
    return housing_prepared_df



def pipeline_transform_features(housing):

    housing_num = housing.drop("ocean_proximity", axis=1)
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    combined_attr_adder = CombinedAttributesAdder(True)
    num_pipeline = Pipeline([
     ('selector', DataFrameSelector(num_attribs)),
     ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
     ('attribs_adder', combined_attr_adder),
     ('std_scaler', StandardScaler()),
     ])

    cat_pipeline = Pipeline([
     ('selector', DataFrameSelector(cat_attribs)),
     ('label_binarizer', SupervisionFriendlyLabelBinarizer()),
     ])


    full_pipeline = FeatureUnion(transformer_list=[
     ("num_pipeline", num_pipeline),
     ("cat_pipeline", cat_pipeline),
     ])


    housing_prepared = full_pipeline.fit_transform(housing)
    one_hot_encoder_cols = list(housing["ocean_proximity"].unique())
    housing_prepared_df = from_array_to_df(housing_prepared, num_attribs, combined_attr_adder, one_hot_encoder_cols )
    return housing_prepared_df







