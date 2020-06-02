from src.pipeline.pipeline_constants import \
    TO_REMOVE_CAPPED_TARGET_PRICE_VALUE, TARGET_PRICE_VALUE_FEATURE_NAME, \
    TO_REMOVE_CAPPED_MEDIAN_AGE, MEDIAN_AGE_FEATURE_NAME
from src.transformers.OutliersRemover import OutliersRemover
from src.transformers.transformers import CappedValuesRemover
import pandas as pd

def _remove_capped_feature(housing: pd.DataFrame, feature: str)-> pd.DataFrame:
    capped_remover_transformer = CappedValuesRemover(feature)
    housing_cleaned = capped_remover_transformer.transform(housing)
    housing_cleaned.reset_index(drop=True, inplace=True)
    return housing_cleaned




def _capped_feature_removal_stage(housing: pd.DataFrame ):
    if TO_REMOVE_CAPPED_TARGET_PRICE_VALUE:
        housing = _remove_capped_feature(housing, TARGET_PRICE_VALUE_FEATURE_NAME)

    if TO_REMOVE_CAPPED_MEDIAN_AGE:
        housing = _remove_capped_feature(housing, MEDIAN_AGE_FEATURE_NAME)

    return housing




def _outliers_removal_stage(housing: pd.DataFrame):
    outline_remover = OutliersRemover()
    housing_cutted = outline_remover.transform(housing)
    housing_cutted.reset_index(drop=True, inplace=True)
    return housing_cutted




def pre_pipeline_district_removal(housing: pd.DataFrame):
    housing = _capped_feature_removal_stage(housing)
    housing = _outliers_removal_stage(housing)
    return housing
