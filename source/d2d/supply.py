## Supply-related associated with FleetMaaS ##
import numpy as np
import math
import random

def work_preday(vehicles, params):
    '''determines which job seekers participate on a given day, a pre-day alternative to the within-day D2D_driver_out function in MaaSSim''' 
    
    for index, row in vehicles.iterrows():
        util_nd = params.evol.drivers.particip.beta * row.res_wage
        expected_income_list = list(map(float,row.expected_income.split(";"))) # income for each platform
        regist_list = row.registered.split(";") # which platforms is the job seeker registered with
        registered = (sum([i == 'True' for i in regist_list]) > 0) # registered with at least one platform
        if row.multihoming:
            expected_income = sum([0 if math.isnan(i) else i for i in expected_income_list]) # income of all platforms together
        else:
            registered_plf = np.array([i == 'True' for i in regist_list])
            if registered_plf.sum() == 1: # registered with a platform
                expected_income = np.array(expected_income_list)[registered_plf][0]
        util_d = params.evol.drivers.particip.beta * expected_income
        prob_d = (np.exp(util_d) / (np.exp(util_d) + np.exp(util_nd))) * registered
        vehicles.loc[index,'ptcp'] = prob_d < random.random()

    return vehicles


def set_multihoming_drivers(driver_df, params):
    '''determines which job seekers are open to multi-home'''
    mh_share = params.evol.drivers.get('mh_share', 1)
    driver_df['multihoming'] = np.random.random(params.nV) < mh_share

    return driver_df
