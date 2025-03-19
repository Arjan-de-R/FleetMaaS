import sys
import os
import logging
MAASSIM_DIR = "MaaSSim"
FLEETPY_DIR = "FleetPy"
sys.path += [MAASSIM_DIR, FLEETPY_DIR]
from simulator_fleetmaas import simulate_parallel
from continue_simulator_fleetmaas import continue_simulate_parallel
from MaaSSim.src_MaaSSim.utils import save_config, get_config
from MaaSSim.src_MaaSSim.d2d_sim import *

### CHOOSE CONFIG FILE
params = get_config(os.path.join('MaaSSim','data','config','AMS_TMC.json'))

### SERVICE TYPES
params.platforms.service_types = ['solo', 'pool']  # list with 'solo' or 'pool' for each platform
# params.evol.travellers.inform.start_pool_detour = 0.25
### OPTIONAL: ADD / CHANGE MODEL PARAMETERS
params.nP = 1000 # travellers
params.nV = 100 # drivers
params.dem_mgmt = 'tmc' # 'None', 'tmc', 'lpr', 'cgp'
params.tmc.duration = 10  # days
params.nD = 1 * params.tmc.duration # max. number of days
params.tmc.allocated_credits_per_day = 10 # credit/day
params.evol.travellers.vot_determination = "from_income"
params.evol.travellers.baseline_log_VoT = -2.75      # Baseline log VoT
params.evol.travellers.income_elasticity = 0.5  # Income elasticity of VoT
params.evol.travellers.random_var_income = 0.5
params.tmc.pref_trading.method = "regression"
params.tmc.pref_trading.regression.constant = 0
params.tmc.pref_trading.regression.balance = -0.25
params.tmc.pref_trading.regression.price = -10
params.tmc.pref_trading.regression.days = 0

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
params.t0 = pd.Timestamp(2023, 6, 13, 16)

# Convergence
# params.convergence.req_steady_days = 5 # X days in a row a change in perceived income of x-day moving average below the convergence factor
# # params.convergence.factor = 0.002
# params.convergence.first_moving_avg = 3
# params.convergence.second_moving_avg = 3

def generate_paths(params):
    # generates graph paths based on city name
    params.paths.G = os.path.join('MaaSSim','data','graphs','{}.graphml'.format(params.city.split(",")[0]))
    params.paths.skim = os.path.join('MaaSSim','data','graphs','{}.csv'.format(params.city.split(",")[0]))
    return params

def sample_space():
    # analysis of behavioural parameters
    space = DotMap()
    space.service_types = [['solo', 'pool']]
    # space.dem_mh_share = [0.5]
    # space.sup_mh_share = [0.5]
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
