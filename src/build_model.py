


from data.get_data_helper import load_housing_data
from data.split_data_train_test import stratified_split
from src.models_lib.initial_model_try import initial_model_try

from src.transformers.pipeline import pipeline_transform_features
from src.transformers.transformers import CappedValuesRemover

if __name__ == '__main__':
    housing = load_housing_data()
    ## using CappedValuesRemover as initial step on the whole dataset since its the target feature.
    capped_remover_transformer = CappedValuesRemover('median_house_value')
    housing2 = capped_remover_transformer.transform(housing)
    housing2.reset_index(drop=True, inplace=True)


    # split into train and test datasets
    train_set, test_set =stratified_split(housing2)
    #from now on all transformations // data clean will be performed on the training set.
    housing = train_set.copy()
    housing = pipeline_transform_features(housing)


    housing_labels = housing["median_house_value"].copy()
    housing = housing.drop("median_house_value", axis=1)

    # housing_labels = train_set[train_set.index.isin(housing.index)]



    # 3 different run options can either turn on/off:

    # 1. run initial predictors try of total 13 options:
    initial_model_try(housing,housing_labels)


    # 2. fine tune a spesific mode
    # fine_tune_top_models.random_forest_hyperparams_grid_search(
    #     housing_transformed, housing_labels)

    # 3. test the best model on the test set
    #test_model(housing_transformed, housing_labels, test_set)


