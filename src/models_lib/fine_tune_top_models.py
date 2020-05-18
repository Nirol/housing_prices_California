from sklearn.ensemble import RandomForestRegressor
import numpy as np

def random_forest_hyperparams_grid_search(housing_df, housing_labels):
    from sklearn.model_selection import GridSearchCV
    param_grid = [
        {'n_estimators': [25, 30, 40, 50], 'max_features': [3,4,5]},
        # {'bootstrap': [False], 'n_estimators': [3, 10,30, 50, 75, 100],
        #  'max_features': [2, 3, 4, 6, 8,10]},
    ]
    forest_reg = RandomForestRegressor()
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                               scoring='neg_mean_squared_error')
    grid_search.fit(housing_df, housing_labels)


    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    feature_importances = grid_search.best_estimator_.feature_importances_
    col_list = list(housing_df.columns)
    features_importance_list = sorted(zip(feature_importances, col_list), reverse=True)

    print(features_importance_list)
    print("a")
