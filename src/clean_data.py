from my_proj.data.split_data_train_test import stratified_split
from my_proj.src.data_transformations import label_enconder_ocean_prox, \
    fill_na_total_bedrooms, one_hot_encode_ocean_prox
from my_proj.src.transformers.combined_attr_transformer import \
    CombinedAttributesAdder
import pandas as pd

from my_proj.src.transformers.pipeline import pipeline_transform_features

## used before the clean data pipeline
def clean_data(housing):


    # sampling while making sure median income splited unbiased in train/test
    train_set, test_set =stratified_split(housing)
    housing = train_set.copy()
    #TODO
    # clean heavy tail dist
    # clean house value capped district as seen in house value // income corr scatter plot







    # after viz exploration and corr between features and between features and target
    # lets seperate the target/label from the other features:

    housing = train_set.drop("median_house_value", axis=1)
    housing_labels = train_set["median_house_value"].copy()







    #after notebook: corrleation and notbook viz train data:
    # as noted in the train set data viz exploration, we have a number of seemingly untelling features:
    # total_rooms // total_bedroom // population by itself doesn't give us much data to estimate the target price

    # we can combine this features in order to create much more telling variables:



    #
    # attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
    # housing_extra_attribs = attr_adder.transform(housing.values)
    # #back to df
    # a = list(housing.columns)
    # new_cols = list(housing.columns) + ["rooms_per_household", "population_per_household"]
    # housing_df = pd.DataFrame(housing_extra_attribs, columns=new_cols)







    #explore corrleations again after adding new features
    #bedrooms per room found to be really informative and corrleated to the house price
    # so is the room per household


    # next step to fill nan total bedrooms:
    # housing_tr = fill_na_total_bedrooms(housing_df)
    #
    # housing_cat = housing["ocean_proximity"]
    # ocean_prox_1hot_encode = one_hot_encode_ocean_prox(housing_cat)
    # full_df = pd.concat([housing_tr,ocean_prox_1hot_encode], axis=1)



    pipeline_transform_features(housing)