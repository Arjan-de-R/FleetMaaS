import pandas as pd
import os

def load_market_indicators(result_path):
    # Load relevant csv's: market shares and credit market indicators
    market_shares = pd.read_csv(os.path.join(result_path,'5_conv-indicators.csv'))
    market_shares.index.name = 'day'

    return market_shares


def load_tmc_indicators(result_path):
    # Load relevant csv's: market shares and credit market indicators
    tmc_indicators = pd.read_csv(os.path.join(result_path,'6_tmc-indicators.csv'))
    tmc_indicators.index.name = 'day'

    return tmc_indicators