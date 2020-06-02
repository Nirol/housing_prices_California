import numpy as np

def random_split_train_test(data, test_ratio):
 np.random.seed(42)
 shuffled_indices = np.random.permutation(len(data))
 test_set_size = int(len(data) * test_ratio)
 test_indices = shuffled_indices[:test_set_size]
 train_indices = shuffled_indices[test_set_size:]
 return data.iloc[train_indices], data.iloc[test_indices]


import hashlib
def test_set_check(identifier, test_ratio, hash):
 return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


def random_split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
 ids = data[id_column]
 in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
 return data.loc[~in_test_set], data.loc[in_test_set]









import numpy as np
def income_into_category(housing):
    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)


def remove_income_category_colmns(strat_train_set, strat_test_set):
 for set in (strat_train_set, strat_test_set):
  set.drop(["income_cat"], axis=1, inplace=True)



from sklearn.model_selection import StratifiedShuffleSplit
#  Startified Shuffle Split to make sure the income category is evenly split.
def stratified_split(housing):
 income_into_category(housing)
 split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
 for train_index, test_index in split.split(housing, housing["income_cat"]):
  strat_train_set = housing.loc[train_index]
  strat_test_set = housing.loc[test_index]
  remove_income_category_colmns(strat_train_set, strat_test_set)

 return strat_train_set, strat_test_set
