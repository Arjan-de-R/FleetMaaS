import sys
import os
import logging
MAASSIM_DIR = "MaaSSim"
FLEETPY_DIR = "FleetPy"
sys.path += [MAASSIM_DIR, FLEETPY_DIR]
from simulator_fleetmaas import simulate_parallel
from MaaSSim.src_MaaSSim.utils import save_config, get_config
from MaaSSim.src_MaaSSim.d2d_sim import *

### CHOOSE CONFIG FILE
params = get_config(os.path.join('MaaSSim','data','config','FM_AMS_cmpt.json'))

### SERVICE TYPES
params.platforms.service_types = ['solo', 'pool'] # list with 'solo' or 'pooling' for each platform

### OPTIONAL: ADD / CHANGE MODEL PARAMETERS
## General settings
# params.city = "Amsterdam, Netherlands"
# params.paths.albatross = os.path.join(MAASSIM_DIR,'data','albatross')
# params.paths.requests = os.path.join(MAASSIM_DIR,'data','demand','{}'.format(params.city.split(",")[0],'albatross','preprocessed.csv')
# params.paths.PT_trips = os.path.join(MAASSIM_DIR,'data','demand','{}'.format(params.city.split(",")[0],'albatross','req_PT.csv')
# params.study_name = 'competition_trb24'
# params.paths.fleetpy_config = 'constant_config.csv'
params.nP = 20000 # travellers
params.nV = 100 # drivers
params.nD = 10 # max. number of days
# params.simTime = 8 # hours
## Platform settings - platform 0
# params.platforms.base_fare = 1.5 #euro
# params.platforms.fare = 1.5 #euro/km
# params.platforms.min_fare = 0 # euro
# params.platforms.comm_rate = 0.25 #rate
# params.platforms.max_wait_time = 600 # maximum time from assignment to pick-up allowed by platform
# params.platforms.match_obj = 'func_key:total_system_time'
# params.platforms.max_rel_detour_pool = 40 # if pooling is not allowed, this is set to 0 (with non-zero additional boarding time preventing pooling for identical trip requests)
params.platforms.pool_discount = (1/3) # relative to solo fare
## Platform settings - platform 1 (if not specified, the same as platform 0)
# params.platforms.fare_1 = 1.5 #euro/km #1.63
# params.platforms.comm_rate_1 = 0.25 #rate
# params.platforms.max_rel_detour_1 = 40 # if pooling is not allowed, this is set to 0 (with non-zero additional boarding time preventing pooling for identical trip requests)
## Multi-homing behaviour
# params.evol.travellers.mh_share = 0.5 # share of travellers open to multi-homing
# params.evol.drivers.mh_share = 0.5 # share of drivers open to multi-homing
params.evol.drivers.particip.beta = 0.05
params.evol.travellers.inform.std_fact = 1.0
params.evol.drivers.inform.std_fact = 1.0
params.evol.drivers.kappa_comm = 0.5
params.evol.drivers.regist.samp = 0.5
params.evol.drivers.regist.min_days = 0
params.evol.travellers.mode_pref.min_wts_constant = -0.554 # used in uniform distribution
params.evol.travellers.mode_pref.ASC_car = -1.96
params.evol.travellers.mode_pref.ASC_pt = -4.14
params.evol.travellers.mode_pref.ASC_rs = -5.18
params.evol.travellers.mode_pref.ASC_bike_sd = 5.75
params.evol.travellers.mode_pref.ASC_car_sd = 3.19
params.evol.travellers.mode_pref.ASC_rs_sd = 1.92
params.evol.travellers.mode_pref.ASC_pt_sd = 1.15
params.evol.travellers.mode_pref.beta_cost = -0.381
params.evol.travellers.mode_pref.ivt_mean_lognorm = 0+np.log(0.0633) # mean of the normal distribution underlying the lognorm distribution
params.evol.travellers.mode_pref.ivt_sigma_lognorm = 1 # standard deviation of the normal distribution underlying the lognorm distribution
params.evol.travellers.min_prob = 0.05
params.alt_modes.pt.base_fare = 1.0
params.alt_modes.pt.km_fare = 0.2

## Starting conditions
params.evol.travellers.inform.prob_start = 0.8
params.evol.travellers.regist = DotMap()
params.evol.travellers.regist.prob_start = 0.8
params.evol.drivers.inform.prob_start = 0.8
params.evol.travellers.regist.samp = 0.5
params.evol.travellers.regist.min_days = 0

params.evol.travellers.inform.beta = 1
## Start time
params.t0 = pd.Timestamp(2023, 6, 13, 9)

# Convergence
params.convergence = DotMap()
params.convergence.req_steady_days = 5 # X days in a row a change in perceived income below the convergence factor
params.convergence.factor = 0.01

def generate_paths(params):
    # generates graph paths based on city name
    params.paths.G = os.path.join('MaaSSim','data','graphs','{}.graphml'.format(params.city.split(",")[0]))
    params.paths.skim = os.path.join('MaaSSim','data','graphs','{}.csv'.format(params.city.split(",")[0]))
    return params

def sample_space():
    # analysis of behavioural parameters
    space = DotMap()
    space.dem_mh_share = [0.6]
    space.sup_mh_share = [0.5]
    space.repl_id = [0]
    return space

def determine_n_threads(search_space):
    '''determine number of threads based on scenario dimensions'''
    n_thread = 1
    # Iterate over the items
    for key, value in search_space.items():
        n_thread = n_thread * len(value)
    return n_thread

if __name__=="__main__":
    search_space=sample_space()
    params.parallel.nThread = determine_n_threads(search_space)

    # # # OPTIONAL: If you want to save the parameter values to a config json
    # params.t0 = params.t0.to_pydatetime().strftime('%Y-%m-%d %H:%M:%S')
    # params.NAME = "FM_AMS_cmpt"
    # params.paths.params = os.path.join(MAASSIM_DIR,"data","config")
    # save_config(params)

    simulate_parallel(params=params, search_space=sample_space())
