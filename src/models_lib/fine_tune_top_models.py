from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import HuberRegressor
import pickle
import pandas as pd

def fine_tune_huberregressor(housing: pd.DataFrame, housing_labels: pd.DataFrame, file_path=None):
    """
    Separate fine tune method for the huberregressor


    :param housing: The dataframe input.
    :type housing: pd.DataFrame
    """
    param_grid = [
        {'epsilon': [1, 1.2,1.35, 1.5, 1.75], 'max_iter': [5000],
         'alpha': [0.0001, 0.0005 , 0.001]}

    ]
    huber = HuberRegressor()
    grid_search = GridSearchCV(huber, param_grid, cv=5, verbose=8,
                               scoring='neg_mean_squared_error')

    grid_search.fit(housing, housing_labels)
    if file_path:

        with open(file_path,
                  'wb') as handle:
            pickle.dump(grid_search, handle)








def _print_feature_importance(housing_df, feature_importance):
    col_list = list(housing_df.columns)
    features_importance_list = sorted(zip(feature_importance, col_list),
                                      reverse=True)
    print(features_importance_list)





def hyperparams_grid_search(housing_df: pd.DataFrame, housing_labels: pd.DataFrame, file_path=None):
    """
    hyperparams grid search method. the method saves the result into disk
    print each parameter combination mean error and std.
    In case the  estimator produce feature importance the method will print it.


    :param housing: The dataframe input.
    :type housing: pd.DataFrame
    """

    # parameter grid for gridsearch, adusted for estimator being used.
    param_grid = [

        {'bootstrap': [False], 'n_estimators': [600],
         'max_features': [ 6], 'min_samples_split': [2,4],
         'min_samples_leaf': [1],
         'max_depth': [60,70,90,115]
         },
    ]

    # The estimator used
    forest_reg = RandomForestRegressor()
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5, verbose=8,
                               scoring='neg_mean_squared_error')
    grid_search.fit(housing_df, housing_labels)



    # save results
    if file_path:
        with open(file_path, 'wb') as handle:
            pickle.dump(grid_search, handle)

    #print the gridsearch combination results:
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    # print feature importance of the best estimator
    if grid_search.best_estimator_.feature_importances_:
        _print_feature_importance(housing_df, grid_search.best_estimator_.feature_importances_)



