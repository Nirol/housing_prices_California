import pandas as pd

from src.project_settings import CLEAN_OUTLIERS_DICT


def _samples_to_remove(isUpperValue, housing,feature, inner_dict):
    series_ = housing[feature]
    if isUpperValue:
        qurantile_value = series_.quantile(inner_dict['upper_qurant_percent'])
        return housing[housing[feature] > qurantile_value]
    else:
        qurantile_value = series_.quantile(inner_dict['lower_qurant_percent'])
        return housing[housing[feature] < qurantile_value]


class OutliersRemover():
    """
    Transformer to remove outliers based on project setting CLEAN_OUTLIERS_DICT.

    Methods
    -------
    transform(housing):
        Transform the dataframe data, returning filtered dataframe.

    """
    def transform(self, housing: pd.DataFrame)-> pd.DataFrame:
        df_empty = pd.DataFrame()
        for inner_dict in CLEAN_OUTLIERS_DICT.values():
            for feature in inner_dict['features']:
                low_filtered_samples =_samples_to_remove(False, housing, feature, inner_dict)
                high_filtered_samples = _samples_to_remove(True, housing,
                                                          feature, inner_dict)
                df_empty = df_empty.append(low_filtered_samples)
                df_empty = df_empty.append(high_filtered_samples)
        housing_clean = pd.concat([housing, df_empty, df_empty]).drop_duplicates(
                        keep=False)
        return housing_clean