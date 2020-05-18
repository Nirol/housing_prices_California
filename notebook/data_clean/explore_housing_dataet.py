from my_proj.data.get_data_helper import load_housing_data




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


#histograms of the different features
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()


# we can evaluate that:
# income values were transformed to 0 - 15 instead of original usd salaries  ( it turns out to be in 10k usd units )
# house median age and income were capped at a certain value, the house value is our predictor target so this might be a proelem
# if it is acceptable for the price prediction to be capped at 500k as well we are fine,
# else, we need to remove the capped house price value samples and hopefully get new labeled samples with price value above the cap.


# more generally regarding all features:
#TODO handle later on
# Each features scale differently  -> to be handeled in feature scaling
#  skew // tail heavy histograms for most features  - detailed on the README file


from scipy import stats
import numpy as np
def drop_numerical_outliers(df, z_thresh=3):
    # Constrains will contain `True` or `False` depending on if it is a value below the threshold.
    constrains = df.select_dtypes(include=[np.number]) \
        .apply(lambda x: np.abs(stats.zscore(x)) < z_thresh, reduce=False) \
        .all(axis=1)
    # Drop (inplace) values set to be rejected
    df.drop(df.index[~constrains], inplace=True)

def evaluate_features_outliers(housing):
    rooms = housing["total_rooms"]
    housing["rooms_z_score"] = stats.zscore(rooms)
    a = housing[["total_rooms", "rooms_z_score"]]
    a = a.sort_values(by ='total_rooms' )

    rooms.quantile(.0025)
    x = rooms[~rooms.between(rooms.quantile(.05), rooms.quantile(.99))]  # without outliers


    #function to automatlicly clean outliers from numerica col, based on z_threshold.:
    #drop_numerical_outliers(housing)

    # we will examine


def explore_top_values(housing):
    bb = housing['median_house_value'].value_counts().sort_values(ascending=False)
    print(bb.nlargest(10))