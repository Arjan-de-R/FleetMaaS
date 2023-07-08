import sys
import os
MAASSIM_DIR = "MaaSSim"
FLEETPY_DIR = "FleetPy"
sys.path += [MAASSIM_DIR, FLEETPY_DIR]
from simulator_fleetmaas import simulate_parallel
from MaaSSim.src_MaaSSim.utils import save_config, get_config
from MaaSSim.src_MaaSSim.d2d_sim import *

### CHOOSE CONFIG FILE
params = get_config(os.path.join('MaaSSim','data','config','Ams_FleetPy.json'))

### OPTIONAL: ADD / CHANGE MODEL PARAMETERS
## General settings
# params.city = "Amsterdam, Netherlands"
# params.paths.albatross = os.path.join(MAASSIM_DIR,'data','albatross')
# params.paths.requests = os.path.join(MAASSIM_DIR,'data','demand','{}'.format(params.city.split(",")[0],'albatross','preprocessed.csv')
# params.paths.PT_trips = os.path.join(MAASSIM_DIR,'data','demand','{}'.format(params.city.split(",")[0],'albatross','req_PT.csv')
# params.study_name = 'competition_trb24'
# params.paths.fleetpy_config = 'constant_config.csv'
# params.nP = 5000 # travellers
# params.nV = 100 # drivers
# params.nS = 2 # number of service providers - more than one currently only works with FleetPy
# params.nD = 3 # days
# params.simTime = 8 # hours
## Platform settings - platform 0
# params.platforms.base_fare = 1.5 #euro
# params.platforms.fare = 1.5 #euro/km
# params.platforms.min_fare = 0 # euro
# params.platforms.comm_rate = 0.25 #rate
# params.platforms.max_wait_time = 600 # maximum time from assignment to pick-up allowed by platform
# params.platforms.match_obj = 'func_key:total_system_time'
# params.platforms.max_rel_detour = 40 # if pooling is not allowed, this is set to 0 (with non-zero additional boarding time preventing pooling for identical trip requests)
## Platform settings - platform 1 (if not specified, the same as platform 0)
# params.platforms.fare_1 = 1.5 #euro/km #1.63
# params.platforms.comm_rate_1 = 0.25 #rate
# params.platforms.max_rel_detour_1 = 40 # if pooling is not allowed, this is set to 0 (with non-zero additional boarding time preventing pooling for identical trip requests)
## Multi-homing behaviour
# params.evol.travellers.mh_share = 1 # share of travellers open to multi-homing
# params.evol.drivers.mh_share = 1 # share of drivers open to multi-homing
## Traveller filter
params.evol.travellers.min_prob = 0
## Start time
# params.t0 = pd.Timestamp(2023, 6, 13, 9)

def generate_paths(params):
    # generates graph paths based on city name
    params.paths.G = os.path.join('MaaSSim','data','graphs','{}.graphml'.format(params.city.split(",")[0]))
    params.paths.skim = os.path.join('MaaSSim','data','graphs','{}.csv'.format(params.city.split(",")[0]))
    return params

def sample_space():
    # analysis of behavioural parameters
    space = DotMap()
    space.comm_rate = [0.25]
    space.repl_id = [0]
    return space

def determine_n_threads(search_space):
    '''determine number of threads based on scenario dimensions'''
    n_thread = 1
    # Iterate over the items
    for key, value in search_space.items():
        n_thread = n_thread * len(value)
    
    return n_thread

search_space=sample_space()
params.parallel.nThread = determine_n_threads(search_space)

## OPTIONAL: If you want to save the parameter values to a config json
# params.t0 = params.t0.to_pydatetime().strftime('%Y-%m-%d %H:%M:%S')
# params.NAME = "Ams_FleetPy"
# params.paths.params = os.path.join(MAASSIM_DIR,"data","config")
# save_config(params)

simulate_parallel(params=params, search_space=sample_space())
