
from data.get_data_helper import load_housing_data
import pandas as pd
def first_glance_over_dataset():


    housing = load_housing_data()

    housing.head()


    housing.info()

    #  Notice that the total_bed
    # rooms attribute has only 20,433 non-null values, meaning that 207 districts are
    # missing this feature. We will need to take care of this later


    # ocean_proximity  is the only text value feature, the rest are numerical
    #lets explore the possible values for ocean_proximity:

    housing["ocean_proximity"].value_counts()

    # we confirmed those are categorial feature with only 5 possible text values overall


    # first look on the numerical feilds values:

    describe_table = housing.describe()
    describe_table.T.to_csv('train_set_stats', index=True)

    #histograms of the different features
    import matplotlib.pyplot as plt
    housing.hist(bins=50, figsize=(15,10))
    plt.show()


    # we can evaluate that:
    # income values were transformed to 0 - 15 instead of original usd salaries  ( it turns out to be in 10k usd units )
    # house median age and income were capped at a certain value, the house value is our predictor target so this might be a proelem
    # if it is acceptable for the price prediction to be capped at 500k as well we are fine,
    # else, we need to remove the capped house price value samples and hopefully get new labeled samples with price value above the cap.




from scipy import stats
import numpy as np
def drop_numerical_outliers(df: pd.DataFrame, z_thresh: int =3):
    # Constrains will contain `True` or `False` depending on if it is a value below the threshold.
    constrains = df.select_dtypes(include=[np.number]) \
        .apply(lambda x: np.abs(stats.zscore(x)) < z_thresh, reduce=False) \
        .all(axis=1)
    z_thresh  = 2

    constrains = df[["total_rooms", "total_bedrooms", "population", "households" ]] \
        .apply(lambda x: np.abs(stats.zscore(x)) < z_thresh) \
        .all(axis=1)

    constrains.value_counts()
    # Drop (inplace) values set to be rejected
    df.drop(df.index[~constrains], inplace=True)




def evaluate_features_outliers(housing: pd.DataFrame):

    CLEAN_OUTLIERS_DICT = {'hard_filter': {'features': ["total_rooms", "total_bedrooms"], 'upper_qurant_percent' : 0.975, 'lower_qurant_percent': 0 },
                           'med_filter': {'features': ["households"], 'upper_qurant_percent' : 0.9733, 'lower_qurant_percent': 0 },
                           'easy_filter': {'features': ["population"],'upper_qurant_percent': 0.969, 'lower_qurant_percent': 0 } }


    df_empty = pd.DataFrame()
    for inner_dict in CLEAN_OUTLIERS_DICT.values():
        for feature in inner_dict['features']:
            series_ = housing[feature]
            qurantile_value_bottom = series_.quantile(inner_dict['lower_qurant_percent'])
            qurantile_value_upper = series_.quantile(inner_dict['upper_qurant_percent'])
            low_filtered_samples = housing[housing[feature] < qurantile_value_bottom]
           # df_empty = df_empty.append(low_filtered_samples)
            high_filtered_samples = housing[housing[feature] > qurantile_value_upper]
            df_empty = df_empty.append(high_filtered_samples)

    hosung_clean = pd.concat([housing, df_empty, df_empty]).drop_duplicates(
        keep=False)

    import matplotlib.pyplot as plt
    hosung_clean[["total_rooms", "total_bedrooms", "households" , "population"]].hist(bins=100, figsize=(8, 4))
    plt.show()

    import matplotlib.pyplot as plt
    housing[["housing_median_age", "median_house_value"]].hist(bins=50, figsize=(8, 4))
    plt.show()
    bb = housing['housing_median_age'].value_counts().sort_values(ascending=False)
    print(bb.nlargest(10))

    import matplotlib.pyplot as plt
    housing[["housing_median_age"]].hist(bins=50, figsize=(6, 4))
    plt.show()




def explore_capped_values(housing: pd.DataFrame):
    df_to_remove = pd.DataFrame()
    to_remove_samples = housing[housing["median_house_value"] == 500001.0]
    df_to_remove = df_to_remove.append(to_remove_samples)

    to_remove_samples2 = housing[housing["housing_median_age"] == 52.0]
    df_to_remove = df_to_remove.append(to_remove_samples2)


def explore_top_values(housing):
    bb = housing['median_house_value'].value_counts().sort_values(ascending=False)
    print(bb.nlargest(10))