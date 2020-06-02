import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from data.get_data_helper import PROJECT_FOLDER


class ModelResult:
    def __init__(self, name, model, scores):
        self.model_name = name
        self.model = model
        self.scores = scores
        self.mean = scores.mean()
        self.std = scores.std()
        self.has_features_importance = False

    def set_feature_importance(self, col_names):
        if hasattr(self.model, 'feature_importances_'):
            self.has_features_importance = True
            self.feature_importance = getattr(self.model, 'feature_importances_')
            self.features_importance_str = sorted(zip(self.feature_importance, col_names), reverse=True)

    def __str__(self):
        str_components = []
        str_components.append("%s " % self.model_name)
        str_components.append(",")

        str_components.append("%s " % self.mean)
        str_components.append(",")
        str_components.append("%s" % self.std)
        if self.has_features_importance:
            print(self.features_importance_str)
        return "".join(str_components)


def _assert_all_finite(X):
    """Like assert_all_finite, but only for ndarray."""
    X = np.asanyarray(X)
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method.
    if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum())
            and not np.isfinite(X).all()):
        raise ValueError("Input contains NaN, infinity"
                         " or a value too large for %r." % X.dtype)


def test_model_on_trainning_set_cv(housing, housing_labels, model, model_name):
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, housing, housing_labels, scoring="neg_mean_squared_error",
                              cv=10)
    if np.any(scores < 0):
        scores = np.sqrt(-scores)
    else:
        scores = np.sqrt(scores)
    return ModelResult(model_name, model,  scores)



def initial_model_try(housing_transformed, housing_labels):
    housing_transformed = housing
    models_dict = {}
    col_list = list(housing_transformed.columns)


    print("1. Linear Rregression")
    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(housing_transformed, housing_labels)
    models_dict["Linear Rregression"] = test_model_on_trainning_set_cv(housing_transformed, housing_labels, lin_reg, "Linear Rregression")


    print("2. DecisionTreeRegressor")
    from sklearn.tree import DecisionTreeRegressor
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(housing_transformed, housing_labels)
    models_dict["DecisionTreeRegressor"] = test_model_on_trainning_set_cv(housing_transformed, housing_labels, tree_reg, "DecisionTreeRegressor")
    models_dict["DecisionTreeRegressor"].set_feature_importance(col_list)



    print("3. RandomForestRegressor")
    from sklearn.ensemble import RandomForestRegressor
    forest_reg = RandomForestRegressor()
    forest_reg.fit(housing_transformed, housing_labels)
    models_dict["RandomForestRegressor"] = test_model_on_trainning_set_cv(housing_transformed, housing_labels, forest_reg, "RandomForestRegressor")
    models_dict["RandomForestRegressor"].set_feature_importance(col_list)
    print(forest_reg.feature_importances_)


    print("4. Lasso")
    from sklearn import linear_model
    lasso = linear_model.Lasso(max_iter=100000, alpha=0.1, tol= 0.01)
    lasso.fit(housing_transformed, housing_labels)
    models_dict["Lasso"] = test_model_on_trainning_set_cv(housing_transformed, housing_labels, lasso, "Lasso")




    print("5. ElasticNet")
    from sklearn import linear_model
    e_net = linear_model.ElasticNet(alpha=0.1)
    e_net.fit(housing_transformed, housing_labels)
    models_dict["ElasticNet"] = test_model_on_trainning_set_cv(housing_transformed, housing_labels, e_net, "ElasticNet")




    print("6. Ridge")
    reg = linear_model.Ridge(alpha=.5)
    reg.fit(housing_transformed, housing_labels)
    models_dict["Ridge"] = test_model_on_trainning_set_cv(housing_transformed, housing_labels, reg, "Ridge")



    print("7. SVR")
    from sklearn import svm
    svr = svm.SVR()
    svr.fit(housing_transformed, housing_labels)
    models_dict["SVR"] = test_model_on_trainning_set_cv(housing_transformed, housing_labels, svr, "SVR")


    print("9. HuberRegressor")
    from sklearn.linear_model import HuberRegressor
    huber = HuberRegressor(max_iter=5000, epsilon=2.5).fit(housing, housing_labels)
    models_dict["HuberRegressor"] = test_model_on_trainning_set_cv(
        housing, housing_labels, huber, "HuberRegressor")

    boolean_outliers_mask = models_dict["HuberRegressor"].model.outliers_
    housing = housing
    housing["outlier_mask"] = boolean_outliers_mask
    housing["price_label"] = housing_labels

    outliers_df3 = housing[housing["outlier_mask"] == 1]
    outliers_stat = outliers_df3.describe()
    outliers_stat.T.to_csv('outliers_df_train_set.csv', index=True)
    print("10 K NN")

    from sklearn.neighbors import KNeighborsRegressor
    neigh = KNeighborsRegressor()
    neigh.fit(housing_transformed, housing_labels)
    models_dict["K NN"] = test_model_on_trainning_set_cv(housing_transformed, housing_labels, neigh, "K NN")

    print("11 KernelRidge")
    from sklearn.kernel_ridge import KernelRidge
    KRR = KernelRidge()
    KRR.fit(housing_transformed, housing_labels)
    models_dict["KRR"] = test_model_on_trainning_set_cv(housing_transformed, housing_labels, KRR, "KRR")

    print("13 SGDR")

    from sklearn.linear_model import SGDRegressor
    SGDR = SGDRegressor(max_iter=1000, tol=1e-3)
    SGDR.fit(housing_transformed, housing_labels)
    models_dict["SGDR"] = test_model_on_trainning_set_cv(housing_transformed, housing_labels, SGDR, "SGDR")
    import pickle
    with open('ml_intro_SGDR.pickle', 'wb') as handle:
                pickle.dump(models_dict, handle)


    print("16 ExtraTreesRegressor")
    from sklearn.ensemble import ExtraTreesRegressor

    etr = ExtraTreesRegressor(n_estimators=100, random_state=0)
    etr.fit(housing_transformed, housing_labels)
    models_dict["etr"] = test_model_on_trainning_set_cv(housing_transformed,housing_labels, etr, "etr")





    from sklearn.neural_network import MLPRegressor
    print("17 MLPRegressor")
    mlp = MLPRegressor(random_state=1, tol=0.001)
    mlp.fit(housing_transformed, housing_labels)
    models_dict["mlp"] = test_model_on_trainning_set_cv(housing_transformed,housing_labels, mlp, "mlp")














    ## consolidate the top 3 performing models to the end of the script, I will
    # run most of the later test cycles only  on this 3

    print("3. RandomForestRegressor")
    from sklearn.ensemble import RandomForestRegressor
    forest_reg = RandomForestRegressor()
    forest_reg.fit(train_set_X, train_set_Y)
    models_dict["RandomForestRegressor"] = test_model_on_trainning_set_cv(train_set_X, train_set_Y, forest_reg, "RandomForestRegressor")
    models_dict["RandomForestRegressor"].set_feature_importance(col_list)
    print(forest_reg.feature_importances_)






    print("16 ExtraTreesRegressor")
    from sklearn.ensemble import ExtraTreesRegressor

    etr = ExtraTreesRegressor(n_estimators=100, random_state=0)
    etr.fit(train_set_X, train_set_Y)
    models_dict["etr"] = test_model_on_trainning_set_cv(train_set_X,train_set_Y, etr, "etr")









    print("18 GradientBoostingRegressor")
    gbr = GradientBoostingRegressor(random_state=1)
    gbr.fit(train_set_X, train_set_Y)
    models_dict["gbr"] = test_model_on_trainning_set_cv(train_set_X,train_set_Y, gbr, "gbr")





def initial_model_test(train_set_X, train_set_Y ):

    models_dict = {}
    col_list = list(train_set_X.columns)


    # print("3. RandomForestRegressor")
    # from sklearn.ensemble import RandomForestRegressor
    # forest_reg = RandomForestRegressor()
    # forest_reg.fit(train_set_X, train_set_Y)
    # models_dict["RandomForestRegressor"] = test_model_on_trainning_set_cv(train_set_X, train_set_Y, forest_reg, "RandomForestRegressor")
    # models_dict["RandomForestRegressor"].set_feature_importance(col_list)
    # print(forest_reg.feature_importances_)


    # print("16 ExtraTreesRegressor")
    # from sklearn.ensemble import ExtraTreesRegressor
    #
    # etr = ExtraTreesRegressor(n_estimators=100, random_state=0)
    # etr.fit(train_set_X, train_set_Y)
    # models_dict["etr"] = test_model_on_trainning_set_cv(train_set_X,train_set_Y, etr, "etr")


    print("18 GradientBoostingRegressor")
    gbr = GradientBoostingRegressor(random_state=1)
    gbr.fit(train_set_X, train_set_Y)
    models_dict["gbr"] = test_model_on_trainning_set_cv(train_set_X,train_set_Y, gbr, "gbr")

    for key in models_dict:
        print(models_dict[key])