



###################################
#### Pre Pipeline Districts Cut ####
###################################

## Capped Feature Pipeline Options
TO_REMOVE_CAPPED_TARGET_PRICE_VALUE =   "True"  # or "False"
TARGET_PRICE_VALUE_FEATURE_NAME ="median_house_value"
TO_REMOVE_CAPPED_MEDIAN_AGE =   "False"
MEDIAN_AGE_FEATURE_NAME = "housing_median_age"


## Outliers PipeLine Options

# Set which Filter to filter and by how much upper quantile and lower quantile.
# Where upper quantile value of 1 remove no high value outliers
# Upper quantile value of 0.975 removes top ( 1 = 0.975) = 0.025 = 2.5% districts
# by feature.



CLEAN_OUTLIERS_DICT = {
    'filter_type_1': {'features': ["population"], 'upper_qurant_percent': 1,
                    'lower_qurant_percent': 0.055}
}











###################################
#### initial pipeline constants ####
###################################
# category attributes
CAT_ATTRIBS = ["ocean_proximity"]

# numerical attributes to impute and then log transform
# (combined in our pipeline since only total_rooms need impute

NUM_ATTRIBS_TO_IMPUTE =["total_rooms"]




NUM_ATTRIBS_ALL = ["longitude","latitude","housing_median_age" ,"total_rooms", "total_bedrooms", "population", "households", "median_income"]


###################################3########
#### Feature Scaling pipeline constants ####
####################################3#######

### LOG TRANSFORM:
TO_LOG_TRANSFORM = "True"
NUM_ATTRIBS_TO_LOG =["total_rooms", "total_bedrooms", "population",
                   "households"]

### SCALE TRANSFORM:
TO_SCALE_FEATURE = "True"
NUM_ATTRIBS_TO_SCALE = ["longitude","latitude","housing_median_age", "median_income"]





###################################
#### Attribute Adder Pipeline ####
###################################


# To tweek or add ned attributes go to the transfomer at path:
#  src/transformers/CombinedAttributesAdder.py
TO_BUILD_NEW_ATTRIBUTS = "True"
SOURCE_ATTRIBS =["total_rooms", "total_bedrooms", "population",
                   "households"]

ADD_BEDROOM_PER_ROOOM = "True"
NEW_ATTRIBUTES = [ "rooms_per_household", "population_per_household",
                                     "bedrooms_per_room"]




################################################
#### Remove Unused Features (Last Pipeline) ####
################################################

#To remove features,  fill names and indx in the next 4 variables:
CAT_FEATURES_TO_REMOVE =  ["NEAR OCEAN", "<1H OCEAN","ISLAND", "NEAR BAY"]


NUM_FEATURES_TO_REMOVE =['total_rooms', 'population', 'total_bedrooms',
                         'households']

FEATURES_TO_REMOVE =[] # NUM_FEATURES_TO_REMOVE + CAT_FEATURES_TO_REMOVE


