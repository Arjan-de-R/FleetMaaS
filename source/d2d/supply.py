## Supply-related associated with FleetMaaS ##
import numpy as np
import math
import random

def sh_init_reg(params, regist):
    '''determine which platform a single-homing agents is registered with at start of simulation, possibly none'''
    primary_plf = np.zeros(len(regist), dtype=bool)
    probabilities = params.platforms.get('start_reg_plf_share', np.ones(len(regist)) / len(regist))
    regist_plf = np.random.choice(np.arange(len(params.platforms.service_types)), p=probabilities)
    primary_plf[regist_plf] = True
    reg = primary_plf * regist
    
    return reg


def work_preday(vehicles, params):
    '''determines which job seekers participate on a given day, a pre-day alternative to the within-day D2D_driver_out function in MaaSSim''' 
    
    vehicles['ptcp'] = None
    vehicles['ptcp'] = vehicles.ptcp.astype(object)
    for index, row in vehicles.iterrows():
        util_nd = params.evol.drivers.particip.beta * row.res_wage
        registered_anywhere = (row.registered.sum() > 0) # registered with at least one platform
        if row.multihoming:
            expected_income = row.expected_income[0] # income of all platforms is the same anyways
        else:
            if registered_anywhere: # registered with a platform
                expected_income = row.expected_income[row.registered][0] # income of one you are registered with
            else:
                expected_income = -999 # not further used (probability is set to 0 later)
        util_d = params.evol.drivers.particip.beta * expected_income
        prob_d = (np.exp(util_d) / (np.exp(util_d) + np.exp(util_nd))) * registered_anywhere
        ptcp = (random.random() < prob_d)
        vehicles.at[index, 'ptcp'] = ptcp * row.registered

    return vehicles


def set_multihoming_drivers(driver_df, params):
    '''determines which job seekers are open to multi-home'''
    mh_share = params.get('sup_mh_share', 1)
    driver_df['multihoming'] = np.random.random(params.nV) < mh_share

    return driver_df