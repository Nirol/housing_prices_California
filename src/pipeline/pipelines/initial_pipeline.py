from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


import pandas as pd
import numpy as np


from src.project_settings import NUM_ATTRIBS_TO_IMPUTE, CAT_ATTRIBS_TO_ONE_HOT_ENCODE, \
    IMPUTE_TYPE
from src.pipeline.pipelines_helper import DataFrameSelector, \
    SupervisionFriendlyLabelBinarizer,  rebuild_df_initial_pipeline


def _initial_pipeline_categorial(housing: pd.DataFrame) -> pd.DataFrame :
    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(CAT_ATTRIBS_TO_ONE_HOT_ENCODE)),
        ('label_binarizer', SupervisionFriendlyLabelBinarizer()),
    ])
    housing_prepared = cat_pipeline.fit_transform(housing)
    return housing_prepared


def _initial_pipeline_imputer(housing: pd.DataFrame) -> pd.DataFrame :
    impute_num_pipeline =  Pipeline([
        ('selector', DataFrameSelector(NUM_ATTRIBS_TO_IMPUTE)),
        ('imputer', SimpleImputer(missing_values=np.nan, strategy=IMPUTE_TYPE)),

    ])
    housing_prepared = impute_num_pipeline.fit_transform(housing)
    return housing_prepared



def initial_pipeline_wrapper(housing_dirty: pd.DataFrame) -> pd.DataFrame :
    """
    The first initial pipeline stage, independently clean the required impute
    for numerical features and one hot encoding for the categorical features.


    Settings from src/project_settings.py:
    CAT_ATTRIBS_TO_ONE_HOT_ENCODE
    NUM_ATTRIBS_TO_IMPUTE
    IMPUTE_TYPE


    :param housing: The dataframe input.
    :type housing: pd.DataFrame
    :return: The dataframe after the filtering.D:
    :rtype: pd.DataFrame
    """
    imputed_numerical = _initial_pipeline_imputer(housing_dirty)
    one_hot_encode_categorial_features = _initial_pipeline_categorial(housing_dirty)
    df_ready = rebuild_df_initial_pipeline(housing_dirty, imputed_numerical, one_hot_encode_categorial_features )
    return df_ready
