


from data.get_data_helper import load_housing_data
from data.split_data_train_test import stratified_split
from src.models_lib import fine_tune_top_models
from src.models_lib.initial_model_try import initial_model_try, \
    initial_model_try_problematic, initial_model_try_2
from src.models_lib.parse_models_results import parse_initial_models_results, \
    parse_initial_models_results2, parse_random_forest_parameter_tunning
from src.models_lib.test_model import test_model

from src.transformers.pipeline import pipeline_transform_features

if __name__ == '__main__':
    housing = load_housing_data()
    # split into train and test datasets
    train_set, test_set =stratified_split(housing)
    #from now on all transformations // data clean will be performed on the training set.
    housing = train_set.copy()
    housing = housing.drop("median_house_value", axis=1)
    housing_labels = train_set["median_house_value"].copy()

    housing_transformed =  pipeline_transform_features(housing)
    # 3 different run options can either turn on/off:

    # 1. run initial predictors try of total 13 options:
    #initial_model_try(housing_transformed,housing_labels)


    # 2. fine tune a spesific mode
    # fine_tune_top_models.random_forest_hyperparams_grid_search(
    #     housing_transformed, housing_labels)

    # 3. test the best model on the test set
    #test_model(housing_transformed, housing_labels, test_set)


