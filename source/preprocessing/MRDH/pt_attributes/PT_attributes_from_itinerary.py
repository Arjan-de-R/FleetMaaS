### Script for determining key public transport trip attributes based on the itineraries queried using OpenTripPlanner (equivalent to function in src_MaaSSim/d2d_demand.py)
import os
import sys
from dotmap import DotMap
import numpy as np
import pandas as pd

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../MaaSSim/src_MaaSSim")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../MaaSSim")))

# from d2d_demand import load_OTP_result
def load_OTP_result(params):
    # loads the attributes of the recommended PT initeraries

    df = pd.read_csv(params.paths.PT_trips, index_col='id')
    df.index.name = 'pax_id'

    # Determine distance for all PT legs
    # Split mode information in separate legs
    legs = df['modes'].str.replace(r'[','')
    legs = legs.str.split(']', expand=True)
    for column in range(len(legs.columns)):
        legs[column] = legs[column].str.lstrip(', ')  # Remove leading commas
        # Set PT distance for walk segments and empty segments to zero (not part of fare calculation), and extract distance for PT legs
        legs[column] = legs[column].fillna('')
        legs[column] = np.where(((legs[column].str.contains('WALK')==True) | (legs[column] == '')), 0, legs[column].str.split(',').str[2])
        legs[column] = legs[column].astype(int)
    # Calculate total PT distance
    legs['PTdistance'] = legs.sum(axis=1, skipna=True)
    df = df.merge(legs['PTdistance'], how='left', left_index=True, right_index=True)
    # Rename fare
    df = df.rename(columns={'fare': 'PTfare'})
    
    return df

input_file_name = "georequests_PT.csv"

params = DotMap()
params.paths.PT_trips = os.path.join(os.path.dirname(os.path.abspath(__file__)), input_file_name) #os.path.join(os.getcwd(), "source", "preprocessing", "MRDH", "pt_attributes", "georequests_PT.csv")
df = load_OTP_result(params)

df.to_csv(os.path.join(os.path.dirname(__file__), 'pt_attributes.csv'))


