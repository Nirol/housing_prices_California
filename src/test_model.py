import numpy as np

def     test_model_on_trainning_set(housing, housing_labels, reg):

    from sklearn.metrics import mean_squared_error
    housing_predictions = reg.predict(housing)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    print(lin_rmse)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


def test_model_on_trainning_set_cv(housing, housing_labels, reg):
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(reg, housing, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
    rmse_scores = np.sqrt(-scores)
    display_scores(rmse_scores)
