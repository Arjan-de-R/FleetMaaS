import numpy as np
from MaaSSim.MaaSSim.d2d_demand import *

def mode_choice_preday(inData, params):
    "determine the mode at the start of a day for a pool of travellers"
    passengers = inData.passengers
    rs_wait = passengers.expected_wait
    
    U_rs = util_rs(inData, params, rs_wait)
    utils = pd.DataFrame({'bike': passengers.U_bike, 'car': passengers.U_car, 'pt': passengers.U_pt, 'rs': U_rs})
    utils.loc[~inData.passengers.registered, 'rs'] = -math.inf
    
    probabilities = mode_probs(utils)
    
    cuml = probabilities.cumsum(axis=1)
    draw = cuml.gt(np.random.random(len(passengers)),axis=0) * 1
    probabilities['decis'] = draw.idxmax(axis="columns")
    passengers['mode_day'] = probabilities.decis
    
    return passengers


def set_multihoming_travellers(trav_df, params):
    '''determines which travellers are open to multi-home'''
    mh_share = params.evol.travellers.get('mh_share', 1)
    trav_df['multihoming'] = np.random.random(trav_df.shape[0]) < mh_share

    return trav_df


def start_regist_travs(trav_df, params):
    '''determines which travellers are registered at the start of the simulation'''
    prob_reg_start = params.evol.travellers.get('prob_reg_start', 1)

    trav_df['registered'] = (np.random.rand(trav_df.shape[0]) < prob_reg_start) & trav_df.informed

    return trav_df