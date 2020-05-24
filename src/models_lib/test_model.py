from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

from notebook.test_error_exploration.explore_test_error import \
    explore_test_error
from src.transformers.pipeline import pipeline_transform_features


def __load_from_pickle():

    import pickle
    with open('src/grid_search_random_forest_remove_params.pickle', 'rb') as handle:
        grid = pickle.load(handle)
        return grid.best_estimator_






def test_model(housing_train_x, housing_train_y, test_set):

    # either load an estimator from the grid search best estimator that was saved to pickle
    # or re-fit an estimator with the desirable parameters
   # estimator = __load_from_pickle()

    estimaor =     RandomForestRegressor(bootstrap=True, max_features=6,  n_estimators=30)
    estimaor.fit(housing_train_x, housing_train_y)
    X_test = test_set.drop("median_house_value", axis=1)
    y_test = test_set["median_house_value"].copy()
    X_test_prepared = pipeline_transform_features(X_test)
    final_predictions = estimaor.predict(X_test_prepared)








    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print(final_rmse)

    explore_test_error(X_test, y_test, final_predictions)
