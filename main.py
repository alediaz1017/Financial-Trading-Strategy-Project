import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import vaderSentiment as vader
import eventregistry as er

from eventregistry import*
er = EventRegistry(api_key = "THE API KEY HERE", allowUseofArchive = False)

microsoftURI = er.getConceptUri("Microsoft")
googleURI = er.getConceptUri("Google")
nvidiaURI = er.getConceptUri("Nvidia")