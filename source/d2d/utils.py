import numpy as np
import pandas as pd
import os
from MaaSSim.src_MaaSSim.d2d_demand import *

def zero_to_nan(indicator):
    indicator = indicator.astype(float)
    indicator[indicator == 0] = np.nan
    return indicator


def save_market_shares(inData, params, result_path, day, travs_summary, drivers_summary, d2d_conv, delay_factor):
    '''Create new dataframe containing (experienced) participation values of all days (and potentially market shares of alternative modes)'''
    travs_summary['requests_mh'] = travs_summary.apply(lambda row: row.requests.sum() > 1, axis=1)
    travs_summary['requests_sh_0'] = travs_summary.apply(lambda row: int(row['requests'][0]) * int(not row['requests_mh']), axis=1)
    drivers_summary['ptcp_mh'] = drivers_summary.apply(lambda row: ((~row.out).sum() > 1), axis=1)
    drivers_summary['ptcp_sh_0'] = drivers_summary.apply(lambda row: int(not row['out'][0]) * int(not row['ptcp_mh']), axis=1)
    conv_indic = pd.DataFrame([{'ptcp_dem_mh': travs_summary['requests_mh'].sum(), 'ptcp_dem_sh_0': travs_summary['requests_sh_0'].sum(), 
                                'ptcp_sup_mh': drivers_summary['ptcp_mh'].sum(), 'ptcp_sup_sh_0': drivers_summary['ptcp_sh_0'].sum()}])
    if inData.platforms.shape[0] > 1: # more than one platform
        conv_indic['ptcp_dem_sh_1'] = travs_summary.apply(lambda row: int(row['requests'][1]) * int(not row['requests_mh']), axis=1).sum()
        conv_indic['ptcp_sup_sh_1'] = drivers_summary.apply(lambda row: int(not row['out'][1]) * int(not row['ptcp_mh']), axis=1).sum()
    for mode in ['bike', 'car', 'pt']:
        conv_indic[mode] = travs_summary.chosen_mode.value_counts()[mode] if mode in travs_summary.chosen_mode.value_counts().index else 0
    if params.tmc:
        conv_indic['not_enough_credit'] = travs_summary.chosen_mode.value_counts()['not_enough_credit'] if 'not_enough_credit' in travs_summary.chosen_mode.value_counts().index else 0
    conv_indic['congestion_factor'] = delay_factor
    d2d_conv = pd.concat([d2d_conv, conv_indic])
    # Create a copy of the csv by adding the last row to the already existing csv
    if day == 0: # include the headers on the first day
        if os.path.exists(os.path.join(result_path,'5_conv-indicators.csv')):
            os.remove(os.path.join(result_path,'5_conv-indicators.csv'))
        conv_indic.to_csv(os.path.join(result_path,'5_conv-indicators.csv'), mode='a', index=False, header=True)
    else:
        conv_indic.to_csv(os.path.join(result_path,'5_conv-indicators.csv'), mode='a', index=False, header=False)

    return d2d_conv