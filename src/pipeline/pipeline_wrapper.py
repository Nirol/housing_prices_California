from data.split_data_train_test import stratified_split
from src.pipeline.pipelines.attribute_adder_pipeline import pipeline_attribute_adder
from src.pipeline.pipelines.feature_scaling_pipeline import \
    feature_scaling_pipeline_wrapper
from src.pipeline.pipelines.initial_pipeline import initial_pipeline_wrapper
from src.pipeline.pipelines.remove_feature_pipeline import \
    pipeline_remove_features


def run_pipeline(housing_data):


    housing_after_initial = initial_pipeline_wrapper(housing_data)
    housing_scaled = feature_scaling_pipeline_wrapper(housing_after_initial)

    scaled_housing_with_added_attr = pipeline_attribute_adder(housing_scaled)
    final_df = pipeline_remove_features(scaled_housing_with_added_attr)

    # df = pd.concat([housing_initial_rdy, numerical_with_added_attr_df],
    #                      axis=1)

    return final_df





def wrapper_pipeline_x(samples_set):
    sample_set_transformed = run_pipeline(samples_set)
    Y_set_labels = sample_set_transformed["median_house_value"].copy()
    X_sample_set_transformed = sample_set_transformed.drop("median_house_value",
                                                       axis=1)
    return X_sample_set_transformed,  Y_set_labels




def main_pipeline_wrapper(housing):


    train_set, test_set =stratified_split(housing)


    train_set_X, train_set_Y = wrapper_pipeline_x(train_set)


    test_set_X, test_set_Y = wrapper_pipeline_x(test_set)

    result_dic = {"train_x": train_set_X,  "train_y": train_set_Y,
                    'test_x': test_set_X, "test_y":test_set_Y }

    return result_dic






