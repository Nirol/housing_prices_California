from my_proj.data.split_data_train_test import stratified_split




#original dataet spread of median income after turning the original income value into categories
housing["income_cat"].value_counts() / len(housing)


#using from sklearn.model_selection import StratifiedShuffleSplit to create train/test sets
# with similar income categories distribution
train_set, test_set = stratified_split(housing)


#making sure the distubution of the different income categories is the same in the train/ test sets as it was in the original datasets:
train_set["income_cat"].value_counts() / len(train_set)
test_set["income_cat"].value_counts() / len(test_set)