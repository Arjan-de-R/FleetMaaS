## Supply-related associated with FleetMaaS ##
import numpy as np

def work_preday(vehicles, params):
    '''determines which job seekers participate on a given day, a pre-day alternative to the within-day D2D_driver_out function in MaaSSim''' 
    util_d = params.evol.drivers.particip.beta * vehicles.expected_income
    util_nd = params.evol.drivers.particip.beta * vehicles.res_wage
    prob_d_reg = np.exp(util_d) / (np.exp(util_d) + np.exp(util_nd))
    decis = (prob_d_reg < np.random.random(params.nV)) * vehicles.registered
    vehicles['ptcp'] = decis

    return vehicles


def set_multihoming_drivers(driver_df, params):
    '''determines which job seekers are open to multi-home'''
    mh_share = params.evol.drivers.get('mh_share', 1)
    driver_df['multihoming'] = np.random.random(params.nV) < mh_share

    return driver_df
