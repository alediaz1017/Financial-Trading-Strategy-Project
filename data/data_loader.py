import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.df_combined = None
        self.df_prices_only = None