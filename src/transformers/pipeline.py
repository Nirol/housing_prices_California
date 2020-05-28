from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelBinarizer, RobustScaler

from sklearn.pipeline import FeatureUnion
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

from src.transformers.transformers import CombinedAttributesAdder, \
    AttributesRemover, OutlinerRemover, CappedValuesRemover


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


class SupervisionFriendlyLabelBinarizer(LabelBinarizer):
    def fit_transform(self, X, y=None):
        return super(SupervisionFriendlyLabelBinarizer, self).fit_transform(X)


#To remove features,  fill names and indx in the next 4 variables:
CAT_FEATURES_TO_REMOVE = [] # ['<1H OCEAN', 'ISLAND', 'NEAR BAY', 'INLAND']
NUM_FEATURES_TO_REMOVE = [] # ['total_rooms', 'population', 'total_bedrooms',
                        # 'households']
NUM_FEATURES_TO_REMOVE_idx = [] #[3, 4, 5, 6]

# nearby ocean feature is last (5) so remove the first 4:
CAT_FEATURES_TO_REMOVE_idx =   [] # [0, 1, 2, 3]


def pipeline_transform_features(housing):
    housing_num = housing.drop("ocean_proximity", axis=1)
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    combined_attr_adder = CombinedAttributesAdder(True)




    num_pipeline = Pipeline([
        # we need all num attributes  for combination
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
        ('attribs_adder', combined_attr_adder),
        # after adding the new combination features, remove the unused one before standartization
      #  ('attribs_remover', AttributesRemover(NUM_FEATURES_TO_REMOVE_idx)),

        #('std_scaler', RobustScaler()),
    ])

    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('label_binarizer', SupervisionFriendlyLabelBinarizer()),
        ('attribs_remover', AttributesRemover(CAT_FEATURES_TO_REMOVE_idx)),

    ])

    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),

    ])

    # cleaning rows transformers are not supported by sktlearn yet


    # using outliers\ capped values remover before the pipeline:
    capped_remover_transformer = CappedValuesRemover('housing_median_age')
    housing_cleaned= capped_remover_transformer.transform(housing)
    housing_cleaned.reset_index(drop=True, inplace=True)


    #the pipeline
    housing_prepared = full_pipeline.fit_transform(housing_cleaned)

    # from np array to dataframe with feature column names
    def __from_array_to_df():
        all_cat_colummns = list(housing["ocean_proximity"].unique())
        cat_cols_left = [item for item in all_cat_colummns if
                         item not in CAT_FEATURES_TO_REMOVE]
        newly_created_attr = combined_attr_adder.get_new_attr_created()
        num_cols_left = [item for item in num_attribs if
                         item not in NUM_FEATURES_TO_REMOVE]
        all_columns_list = num_cols_left + newly_created_attr + cat_cols_left
        housing_prepared_df = pd.DataFrame(housing_prepared,
                                           columns=all_columns_list)
        return housing_prepared_df

    housing_prepared_df = __from_array_to_df()

    return housing_prepared_df
