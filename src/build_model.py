from my_proj.data.get_data_helper import load_housing_data
from my_proj.data.split_data_train_test import random_split_train_test_by_id, \
    stratified_split
from my_proj.src.clean_data import  clean_data
from my_proj.src.models_lib import fine_tune_top_models
from my_proj.src.models_lib.initial_model_try import initial_model_try
from my_proj.src.test_model import test_model_on_trainning_set, \
    test_model_on_trainning_set_cv
from my_proj.src.transformers.pipeline import pipeline_transform_features
import numpy as np
if __name__ == '__main__':
    housing = load_housing_data()
    # split into train and test datasets
    train_set, test_set =stratified_split(housing)
    #from now on all transformations // data clean will be performed on the training set.
    housing = train_set.copy()
    housing = housing.drop("median_house_value", axis=1)
    housing_labels = train_set["median_house_value"].copy()

    housing_transformed =  pipeline_transform_features(housing)
    evaluate_features_outliers(housing)
   # initial_model_try(housing_transformed,housing_labels)

    fine_tune_top_models.random_forest_hyperparams_grid_search(housing_transformed, housing_labels)