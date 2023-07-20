import numpy as np
from MaaSSim.src_MaaSSim.d2d_demand import *
from source.d2d.utils import *
import random

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


def start_regist_travs(inData, params):
    '''determines which travellers are registered at the start of the simulation'''
    trav_df = inData.passengers

    def array_with_pref_platform(params):
        array = np.full(len(params.platforms.service_types), False)
        rand_plf = random.randint(0, len(params.platforms.service_types)-1) # preferred platform for single-homers
        array[rand_plf] = True
        return array
    
    prob_reg_start = params.evol.travellers.regist.get('prob_start', 1)
    trav_df['ttrav'] = inData.requests.ttrav.dt.total_seconds()
    trav_df['registered'] = (np.random.rand(trav_df.shape[0]) < prob_reg_start) & trav_df.informed
    trav_df['registered'] = trav_df.apply(lambda row: np.full(len(params.platforms.service_types), True) * row.registered, axis=1)
    trav_df['registered'] = trav_df.apply(lambda row: row.registered if row.multihoming else row.registered * array_with_pref_platform(params), axis=1)
    trav_df['expected_wait'] = trav_df.apply(lambda row: zero_to_nan(row.registered * np.ones(len(inData.platforms.index))) * params.evol.travellers.inform.start_wait, axis=1)
    trav_df['expected_ivt'] = trav_df.apply(lambda row: zero_to_nan(row.registered * row.ttrav), axis=1)
    km_fare = inData.platforms.fare.values
    trav_df['expected_km_fare'] = trav_df.apply(lambda row: zero_to_nan(row.registered * km_fare.mean()) if row.multihoming else zero_to_nan(row.registered * km_fare), axis=1)
    trav_df = trav_df.drop(columns=['ttrav'])
    trav_df['days_since_reg'] = trav_df.apply(lambda row: 0 if row.registered.sum() > 0 else np.nan, axis=1)

    return trav_df


def learn_demand(inData, params, zones, perc_demand):
    '''learn demand based on last demand and previous expected demand, used for repositioning'''
    day_demand = inData.passengers[inData.passengers.mode_day == 'rs']
    reqs_per_zone = day_demand.zone_id.value_counts()
    reqs_per_zone = reqs_per_zone.reindex(zones.zone_id.values, fill_value=0)
    perc_demand['requests'] = (1-params.evol.travellers.kappa) * perc_demand['requests'] + params.evol.travellers.kappa * reqs_per_zone

    return perc_demand