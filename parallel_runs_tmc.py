import sys
import os
import logging
import numpy as np
MAASSIM_DIR = "MaaSSim"
FLEETPY_DIR = "FleetPy"
sys.path += [MAASSIM_DIR, FLEETPY_DIR]
from simulator_fleetmaas import simulate_parallel
from continue_simulator_fleetmaas import continue_simulate_parallel
from MaaSSim.src_MaaSSim.utils import save_config, get_config
from MaaSSim.src_MaaSSim.d2d_sim import *

### CHOOSE CONFIG FILE
params = get_config(os.path.join('MaaSSim','data','config','MRDH_TMC.json'))

### SERVICE TYPES
params.platforms.service_types = ['solo', 'pool']  # list with 'solo' or 'pool' for each platform
# params.evol.travellers.inform.start_pool_detour = 0.25
### OPTIONAL: ADD / CHANGE MODEL PARAMETERS
params.nP = 5000 # travellers
params.nV = 100 # max. number of drivers
params.dem_mgmt = 'tmc' # 'None', 'tmc', 'lpr', 'cgp'
params.tmc.allocated_credits_per_day = 10 # credit/day
params.evol.travellers.vot_determination = "from_income"
params.evol.travellers.baseline_log_VoT = -2.75      # Baseline log VoT
params.evol.travellers.income_elasticity = 0.5  # Income elasticity of VoT
params.evol.travellers.random_var_income = 0.5
params.tmc.pref_trading.method = "regression"
params.tmc.pref_trading.beta_constant = 0
params.tmc.pref_trading.beta_balance = -1
params.tmc.pref_trading.beta_price = 0
params.tmc.pref_trading.beta_days = 0
# params.tmc.pref_trading.sd_beta_constant = 0.3
params.tmc.pref_trading.sd_beta_balance = 0
params.tmc.pref_trading.sd_beta_price = 0
params.tmc.pref_trading.sd_error_term = 2
params.tmc.pref_trading.reference = "perceived_need"
params.evol.travellers.tmc.perc_credit_price_start = 0.25
params.network_type = "FleetPy"
params.speeds.bike = 15/3.6  # used to overwrite ttrav_bike
params.congestion.update_interval = 0.5
params.congestion.uncongested_background_travel_time = 100000 
params.convergence.ttf = 0.01
params.convergence.moving_average_window = 5
params.convergence.stable_iters = 3
params.convergence.error_modal_split = 0.002
params.evol.travellers.start_wait = 120
params.evol.drivers.number_of_drivers_per_trav_solo = 6
params.evol.drivers.number_of_drivers_per_trav_pool = 4
params.convergence.start_pool_detour = 0.2
params.evol.travellers.min_kappa = 0.2
params.city = "Delft"
params.total_road_dist = 100
params.paths.requests = "MaaSSim/data/demand/Delft/requests.csv"
params.paths.passengers = "MaaSSim/data/demand/Delft/passengers.csv"
params.paths.PT_trips = "MaaSSim/data/demand/Delft/req_PT.csv"
params.paths.ttfs = "MaaSSim/data/congestion/Delft/ttfs.csv"
params.warmup = 1800
params.cooldown = 1800

# params.tmc.beta_monetary = -0.2
# params.tmc.max_balance = 1000
# params.tmc.credit_mode.bike.base = 1
# params.tmc.credit_mode.car.base = 8
# # params.tmc.credit_mode.pt.base = 1
# params.tmc.credit_mode.solo.base = 12
# params.tmc.credit_mode.pool.base = 8
# params.tmc.credit_mode.solo.dist_add_center = 9
# params.tmc.credit_mode.pool.dist_add_center = 6
# params.tmc.credit_mode.car.dist_add_center = 6
# params.zone_charge.car=30
# params.zone_charge.solo=30
# params.charging_scheme = "add_centre_charge"

# params.city_charge.solo = 20
# params.city_charge.pool = 0
# params.city_charge.car = 20

# params.tmc.allocated_credits_per_day = [15, 12.5, 10, 7.5, 5]

## Start time
params.t0 = pd.Timestamp(2025, 4, 1, 7)

def generate_paths(params):
    # generates graph paths based on city name
    params.paths.G = os.path.join('MaaSSim','data','graphs','{}.graphml'.format(params.city.split(",")[0]))
    params.paths.skim = os.path.join('MaaSSim','data','graphs','{}.csv'.format(params.city.split(",")[0]))
    return params

def sample_space():
    # analysis of behavioural parameters
    space = DotMap()
    space.service_types = [['solo', 'pool']]
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
    # continue_simulate_parallel(params=params, search_space=sample_space())
