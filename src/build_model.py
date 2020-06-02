from data.get_data_helper import load_housing_data
from src.models_lib import fine_tune_top_models
from src.models_lib.initial_model_try import  initial_model_test
from src.models_lib.test_model import test_model
from src.pipeline.pipeline_wrapper import main_pipeline_wrapper
from src.transformers.DistrictsRemoval import pre_pipeline_district_removal


if __name__ == '__main__':
    housing = load_housing_data()

    housing = pre_pipeline_district_removal(housing)

    train_test_result_dict = main_pipeline_wrapper(housing)




    # 3 different run options can either turn on/off:

    # 1. run initial predictors try of total 13 options:
    initial_model_test(train_test_result_dict["train_x"],train_test_result_dict["train_y"])


    # 2. fine tune a spesific mode
    fine_tune_top_models.hyperparams_grid_search( train_test_result_dict["train_x"], train_test_result_dict["train_y"])

    # 3. test the best model on the test set
    test_model(train_test_result_dict["train_x"],
               train_test_result_dict["train_y"],
               train_test_result_dict["test_x"],
               train_test_result_dict["test_y"])


