from src.project_settings import FEATURES_TO_REMOVE
import pandas as pd

def pipeline_remove_features(housing: pd.DataFrame)->  pd.DataFrame:
    """
    Last pipeline stage to remove features we dont need at the end of the
    pipeline data cleaning.


    Settings from src/project_settings.py:
    FEATURES_TO_REMOVE


    :param housing: The dataframe input.
    :type housing: pd.DataFrame
    :return: The dataframe after the filtering.D:
    :rtype: pd.DataFrame
    """
    if FEATURES_TO_REMOVE:
        housing.drop(columns=FEATURES_TO_REMOVE, inplace=True, axis=1)
    return housing


