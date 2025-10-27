import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import vaderSentiment as vader
from datetime import datetime, timedelta 

## Import historical stock ticker data from Yahoo Finance
## Download the functional libaries if not already
## pip install yfinance pandas matplotlib numpy vadersentiment

ticker = "AAPL"
ticker = "MSFT"
ticker = "NVDA"

## Include extra information above on the stock ticker analyzation start date and possible end date
## Begin to also retrieve new article headlines and informaiton on deals done between the designated companies
