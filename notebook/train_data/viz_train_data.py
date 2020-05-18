


# going to look only on the train dataset from now on:
housing = train_set.copy()
#  scatter by long/latt

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

import matplotlib.pyplot as plt


#color for price
#radius for population

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
 s=housing["population"]/100, label="population",
 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()


#
# This image tells you that the housing prices are very much related to the location
# (e.g., close to the ocean) and to the population density, as you probably knew already.
# It will probably be useful to use a clustering algorithm to detect the main clusters, and
# add new features that measure the proximity to the cluster centers. The ocean prox‐
# imity attribute may be useful as well, although in Northern California the housing
# prices in coastal districts are not too high, so it is not a simple rule.

