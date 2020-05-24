from sklearn.ensemble import RandomForestRegressor
import numpy as np







def print_feature_importance(housing_df, feature_importance):
    col_list = list(housing_df.columns)
    features_importance_list = sorted(zip(feature_importance, col_list),
                                      reverse=True)
    print(features_importance_list)





def random_forest_hyperparams_grid_search(housing_df, housing_labels):
    from sklearn.model_selection import GridSearchCV
    param_grid = [
         {'bootstrap': [False, True], 'n_estimators': [600 ],
          'max_features': [None, 6], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [2, 4]},


        # { 'max_depth'=[None], 'min_samples_split'=[2] , 'max_features'=[None]},


    ]


    forest_reg = RandomForestRegressor()
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5, verbose=8,
                               scoring='neg_mean_squared_error')
    grid_search.fit(housing_df, housing_labels)

    import pickle
    with open('src/grid_search_random_forest_remove_params.pickle', 'wb') as handle:
        pickle.dump(grid_search, handle)

    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)


    print_feature_importance(housing_df, grid_search.best_estimator_.feature_importances_)


