from my_proj.src.test_model import test_model_on_trainning_set, \
    test_model_on_trainning_set_cv






def initial_model_try(housing_transformed, housing_labels):



    print("linear reg:")
    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(housing_transformed, housing_labels)
    test_model_on_trainning_set_cv(housing_transformed, housing_labels, lin_reg)

    print("descision tree:")
    from sklearn.tree import DecisionTreeRegressor
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(housing_transformed, housing_labels)
    test_model_on_trainning_set_cv(housing_transformed, housing_labels, tree_reg)




    print("random forests:")
    from sklearn.ensemble import RandomForestRegressor
    forest_reg = RandomForestRegressor()
    forest_reg.fit(housing_transformed, housing_labels)
    test_model_on_trainning_set_cv(housing_transformed, housing_labels, forest_reg)

