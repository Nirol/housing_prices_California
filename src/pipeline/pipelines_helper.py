from typing import List

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
from src.project_settings import CAT_ATTRIBS_TO_ONE_HOT_ENCODE, NUM_ATTRIBS_TO_IMPUTE, \
    NUM_ATTRIBS_TO_LOG, NUM_ATTRIBS_TO_SCALE, TO_LOG_TRANSFORM, \
    TO_SCALE_FEATURE, TO_BUILD_NEW_ATTRIBUTS, SOURCE_ATTRIBS, NEW_ATTRIBUTES










class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select only the required  features for the pipeline cleaning
     process out of the df.

    Methods
    -------
    transform(housing):
        Transform the dataframe data, returning filtered dataframe.

    """
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame)->  pd.DataFrame:

        return X[self.attribute_names]



class SupervisionFriendlyLabelBinarizer(LabelBinarizer):
    """
    Transformer wrapping LabelBinarizer so it receive df input conviently as part
    of sklearn pipeline.

    Methods
    -------
    fit_transform(housing):
        Transform the dataframe data, returning one hot encoding categorl  dataframe.

    """
    def fit_transform(self, X, y=None):
        return super(SupervisionFriendlyLabelBinarizer, self).fit_transform(X)


def _create_cat_attributes_list(housing: pd.DataFrame) -> List[str]:
    all_cat_colummns = []
    for cat_attr in CAT_ATTRIBS_TO_ONE_HOT_ENCODE:
        curr_list = list(housing[cat_attr].unique())
        all_cat_colummns = all_cat_colummns + curr_list
    return all_cat_colummns



def rebuild_df_initial_pipeline(dirty_housing,log_num_feature, categorial_features):
    housing_copy = dirty_housing.copy()
    all_cat_colummns = _create_cat_attributes_list(housing_copy)


    housing_copy.drop(NUM_ATTRIBS_TO_IMPUTE,inplace=True, axis=1)
    housing_copy.reset_index(drop=True, inplace=True)
    housing_copy.drop(CAT_ATTRIBS_TO_ONE_HOT_ENCODE, inplace=True, axis=1)
    housing_copy.reset_index(drop=True, inplace=True)
    mid_df=pd.concat([housing_copy, pd.DataFrame(data = log_num_feature, columns=NUM_ATTRIBS_TO_IMPUTE)], axis=1)
    final_df = pd.concat([mid_df, pd.DataFrame(data = categorial_features, columns=all_cat_colummns)], axis=1)

    return final_df










def rebuild_df_feature_scaling(dirty_housing,log_scale_features, scale_features):
    housing_copy = dirty_housing.copy()

    if TO_LOG_TRANSFORM:
        housing_copy.drop(NUM_ATTRIBS_TO_LOG,inplace=True, axis=1)
        housing_copy.reset_index(drop=True, inplace=True)
        mid_df = pd.concat([housing_copy, pd.DataFrame(data=log_scale_features,
                                                       columns=NUM_ATTRIBS_TO_LOG)],
                           axis=1)
    else:
        mid_df = housing_copy
    if TO_SCALE_FEATURE:
        mid_df.drop(NUM_ATTRIBS_TO_SCALE, inplace=True, axis=1)
        mid_df.reset_index(drop=True, inplace=True)
        final_df = pd.concat([mid_df, pd.DataFrame(data = scale_features, columns=NUM_ATTRIBS_TO_SCALE)], axis=1)

    else:
        final_df=mid_df
    return final_df






def rebuild_df_attribute_adder(housing, housing_added_attribute):
    housing_copy = housing.copy()
    if TO_BUILD_NEW_ATTRIBUTS:
        housing_added_attribute.drop(SOURCE_ATTRIBS, inplace=True, axis=1)
        housing_added_attribute.reset_index(drop=True, inplace=True)
        mid_df = pd.concat([housing_copy, pd.DataFrame(data=housing_added_attribute,
                                                       columns=NEW_ATTRIBUTES)],
                           axis=1)
        return mid_df
    return housing