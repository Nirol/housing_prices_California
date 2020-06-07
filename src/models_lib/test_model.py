from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
import numpy as np


def __load_from_pickle():

    import pickle
    with open('src/grid_search_random_forest_remove_params.pickle', 'rb') as handle:
        grid = pickle.load(handle)
        return grid.best_estimator_






def test_model(train_set_X, train_set_Y, test_set_X, test_set_Y):

    # either load an estimator from the grid search best estimator that was saved to pickle
    # or re-fit an estimator with the desirable parameters
   # estimator = __load_from_pickle()

    estimaor = RandomForestRegressor(bootstrap=False, max_depth=70, max_features=6,
                        min_samples_leaf=1, min_samples_split=2,
                        n_estimators=600)




    estimaor.fit(train_set_X, train_set_Y)



    final_predictions = estimaor.predict(test_set_X)

    final_mse = mean_squared_error(test_set_Y, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print(final_rmse)

    #explore_test_error(X_test_prepared, y_test, final_predictions)
