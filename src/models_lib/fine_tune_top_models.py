from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import numpy as np



def fine_tune_HuberRegressor():
    from sklearn.model_selection import GridSearchCV
    param_grid = [
        {'epsilon': [1, 1.2,1.35, 1.5, 1.75], 'max_iter': [5000],
         'alpha': [0.0001, 0.0005 , 0.001]}


        # { 'max_depth'=[None], 'min_samples_split'=[2] , 'max_features'=[None]},

    ]
    from sklearn.linear_model import HuberRegressor
    huber = HuberRegressor()
    grid_search = GridSearchCV(huber, param_grid, cv=5, verbose=8,
                               scoring='neg_mean_squared_error')

    grid_search.fit(housing, housing_labels)
    import pickle
    with open('src/grid_huber.pickle',
              'wb') as handle:
        pickle.dump(grid_search, handle)








def print_feature_importance(housing_df, feature_importance):
    col_list = list(housing_df.columns)
    features_importance_list = sorted(zip(feature_importance, col_list),
                                      reverse=True)
    print(features_importance_list)





def hyperparams_grid_search(housing_df, housing_labels):
    from sklearn.model_selection import GridSearchCV
    param_grid = [
         { 'n_estimators': [600, 125, 500 , 200],
          'max_features': [ "auto", "sqrt", "log2", 6], 'min_samples_split': [2,6,10], 'min_samples_leaf': [1, 4],
          'max_depth' : [20]
          },
    ]


    forest_reg = ExtraTreesRegressor()
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5, verbose=8,
                               scoring='neg_mean_squared_error')
    grid_search.fit(housing_transformed, housing_labels)


    # save results
    import pickle
    with open('src/grid_search_extra_trees_v3_no_capp.pickle', 'wb') as handle:
        pickle.dump(grid_search, handle)

    # print final result for each parameter combination
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    # print feature importance of the best estimator
    print_feature_importance(housing_transformed, grid_search.best_estimator_.feature_importances_)


