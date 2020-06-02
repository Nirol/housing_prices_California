from sklearn.pipeline import Pipeline
import pandas as pd
from src.pipeline.pipeline_constants import SOURCE_ATTRIBS, \
    TO_BUILD_NEW_ATTRIBUTS, ADD_BEDROOM_PER_ROOOM
from src.pipeline.pipelines_helper import DataFrameSelector, \
    rebuild_df_attribute_adder
from src.transformers.CombinedAttributesAdder import CombinedAttributesAdder


def pipeline_attribute_adder(housing: pd.DataFrame)-> pd.DataFrame:
    combined_attr_adder = CombinedAttributesAdder(ADD_BEDROOM_PER_ROOOM)

    attribute_adder_pipeline = Pipeline([
        # we need all num attributes  for combination
        ('selector', DataFrameSelector(SOURCE_ATTRIBS)),
        ('attribs_adder', combined_attr_adder),

    ])

    if TO_BUILD_NEW_ATTRIBUTS:
        housing_prepared = attribute_adder_pipeline.fit_transform(housing)

        # housing_prepared.drop(SOURCE_ATTRIBS, inplace=True, axis=1)
        # housing_prepared.reset_index(drop=True, inplace=True)
        df_ready = rebuild_df_attribute_adder(housing, housing_prepared)
        return df_ready
    else:
        return  housing


