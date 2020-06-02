from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


import pandas as pd
import numpy as np


from src.pipeline.pipeline_constants import NUM_ATTRIBS_TO_IMPUTE, CAT_ATTRIBS
from src.pipeline.pipelines_helper import DataFrameSelector, \
    SupervisionFriendlyLabelBinarizer,  rebuild_df_initial_pipeline


def _initial_pipeline_categorial(housing: pd.DataFrame) -> pd.DataFrame :
    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(CAT_ATTRIBS)),
        ('label_binarizer', SupervisionFriendlyLabelBinarizer()),
    ])
    housing_prepared = cat_pipeline.fit_transform(housing)
    return housing_prepared


def _initial_pipeline_imputer(housing: pd.DataFrame) -> pd.DataFrame :
    impute_num_pipeline =  Pipeline([
        ('selector', DataFrameSelector(NUM_ATTRIBS_TO_IMPUTE)),
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),

    ])
    housing_prepared = impute_num_pipeline.fit_transform(housing)
    return housing_prepared



def initial_pipeline_wrapper(housing_dirty: pd.DataFrame) -> pd.DataFrame :
    imputed_numerical = _initial_pipeline_imputer(housing_dirty)
    one_hot_encode_categorial_features = _initial_pipeline_categorial(housing_dirty)
    df_ready = rebuild_df_initial_pipeline(housing_dirty, imputed_numerical, one_hot_encode_categorial_features )
    return df_ready
