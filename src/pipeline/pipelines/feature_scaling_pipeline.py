from sklearn.pipeline import Pipeline
import pandas as pd


from src.project_settings import NUM_ATTRIBS_TO_LOG,  \
    TO_LOG_TRANSFORM, NUM_ATTRIBS_TO_SCALE, TO_SCALE_FEATURE
from src.pipeline.pipelines_helper import DataFrameSelector, \
    rebuild_df_feature_scaling
from src.transformers.transformers import FeatureScaleTransformer




def _pipeline_feature_scaling(housing: pd.DataFrame) ->  pd.DataFrame:
    sacle_num_pipeline =  Pipeline([
        ('selector', DataFrameSelector(NUM_ATTRIBS_TO_SCALE)),
         ('scale_transform',FeatureScaleTransformer(transformer = "scale") ),
    ])
    housing_prepared = sacle_num_pipeline.fit_transform(housing)
    return housing_prepared




def _pipeline_feature_scaling_log(housing: pd.DataFrame) ->  pd.DataFrame:
    log_num_pipeline =  Pipeline([
        ('selector', DataFrameSelector(NUM_ATTRIBS_TO_LOG)),
        # transform only the selected log_num_attribs and add them back to the front of the array!
         ('log_transform',FeatureScaleTransformer(transformer = "log") ),
    ])
    housing_prepared = log_num_pipeline.fit_transform(housing)
    return housing_prepared



def feature_scaling_pipeline_wrapper(housing: pd.DataFrame) ->  pd.DataFrame :
    """
    Feature scaling pipeline stage. The pipeline independently handles dataframe
    columns that need to be standrtize scale and log transformed.

    Settings from src/project_settings.py:

    TO_LOG_TRANSFORM, TO_SCALE_FEATURE are binary flags to determine if scaling
    need to be done.
    NUM_ATTRIBS_TO_SCALE, NUM_ATTRIBS_TO_LOG are the feature lists to be scaled.


    :param housing: The dataframe input.
    :type housing: pd.DataFrame
    :return: The dataframe after the filtering.D:
    :rtype: pd.DataFrame
    """
    if TO_LOG_TRANSFORM:
        scaled_data_log = _pipeline_feature_scaling_log(housing)
    if TO_SCALE_FEATURE:
        scaled_data = _pipeline_feature_scaling(housing)

    df_ready = rebuild_df_feature_scaling(housing, scaled_data_log, scaled_data )
    return df_ready
