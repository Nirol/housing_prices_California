


corr_matrix = housing.corr()

corr_matrix["median_house_value"].sort_values(ascending=False)

attributes = ["median_house_value", "rooms_per_household", "bedrooms_per_room",
 "population_per_household"]
corr_matrix = housing[attributes].corr()

corr_matrix["median_house_value"].sort_values(ascending=False)

scatter_matrix(housing[attributes], figsize=(12, 8))


#value which is the target feature in high corr with latitude and age/
#-0.14272 since as we closer to the ocean, smaller latitude values and higher house price
# 0.11411 with age not sure why


#more signifacnt corr:
#median income with negative corr -0.11136 to housing age
#strong positive between total rooms and median income

# very strong corrleation between the grp of total rooms/bedrooms/pop/housholds
#we might be able to remove some of this features



#looking on corrleation scatter plots of features in corr with the house value

from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
 "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))


#income the strongest corr to house value:
housing.plot(kind="scatter", x="median_income", y="median_house_value",
 alpha=0.1)

# This plot reveals a few things. First, the correlation is indeed very strong; you can
# clearly see the upward trend and the points are not too dispersed. Second, the price
# cap that we noticed earlier is clearly visible as a horizontal line at $500,000. But this
# plot reveals other less obvious straight lines: a horizontal line around $450,000,
# another around $350,000, perhaps one around $280,000, and a few more below that.
# You may want to try removing the corresponding districts to prevent your algorithms
# from learning to reproduce these data quirks.