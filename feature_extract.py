import pandas as pd
import os
import numpy as np
import gc
import lightgbm as lgb
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime as dt

from feature_extract_helper import *
#
# split_train_to_hdf()
# set_time_features()
set_clicks_per_quarter_features()