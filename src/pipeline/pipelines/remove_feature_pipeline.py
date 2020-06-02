from src.pipeline.pipeline_constants import FEATURES_TO_REMOVE


def pipeline_remove_features(housing):
    if FEATURES_TO_REMOVE:
        housing.drop(columns=FEATURES_TO_REMOVE, inplace=True, axis=1)
    return housing


