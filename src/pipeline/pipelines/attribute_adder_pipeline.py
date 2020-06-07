from sklearn.pipeline import Pipeline
import pandas as pd
from src.project_settings import SOURCE_ATTRIBS, \
    TO_BUILD_NEW_ATTRIBUTS, ADD_BEDROOM_PER_ROOOM
from src.pipeline.pipelines_helper import DataFrameSelector, \
    rebuild_df_attribute_adder
from src.transformers.CombinedAttributesAdder import CombinedAttributesAdder





def pipeline_attribute_adder(housing: pd.DataFrame)-> pd.DataFrame:
    """
    A pre pipeline stage to remove district rows from the dataframe.
    The function wrap the removal of both capped feature values and extreme
    outliers values. Settings for removal in src/project_settings.py


    :param housing: The dataframe input.
    :type housing: pd.DataFrame
    :return: The dataframe after the filtering.D:
    :rtype: pd.DataFrame
    """
    combined_attr_adder = CombinedAttributesAdder(ADD_BEDROOM_PER_ROOOM)

    attribute_adder_pipeline = Pipeline([
        # we need all num attributes  for combination
        ('selector', DataFrameSelector(SOURCE_ATTRIBS)),
        ('attribs_adder', combined_attr_adder),

    ])

    if TO_BUILD_NEW_ATTRIBUTS:
        housing_prepared = attribute_adder_pipeline.fit_transform(housing)


        df_ready = rebuild_df_attribute_adder(housing, housing_prepared)
        return df_ready
    else:
        return  housing


