import os
import tarfile
from six.moves import urllib




DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"


PROJECT_FOLDER = "C:\\Users\\owner\\PycharmProjects\\handson-ml\\"
_HOUSING_DATA_PATH = PROJECT_FOLDER + "my_proj\\data\\datasets\\housing\\housing.csv"

HOUSING_PATH = os.path.join("my_proj","data","datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()




import pandas as pd
def load_housing_data(housing_path=HOUSING_PATH):

 csv_path = os.path.join(_HOUSING_DATA_PATH)
 return pd.read_csv(csv_path)

import numpy as np


