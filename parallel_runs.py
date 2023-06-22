import sys
from simulator_fleetmaas import simulate_parallel, simulate
from MaaSSim.MaaSSim.utils import save_config, get_config, load_G, generate_demand
from MaaSSim.MaaSSim.d2d_sim import *
sys.path.append('..')

params = get_config('data/config/ams_2B.json')
params.parallel.nThread = 1
params.parallel.nReplications = 1
params.paths.G = 'data/graphs/Amsterdam.graphml'
params.paths.skim = 'data/graphs/Amsterdam.csv'
params.paths.albatross = 'data/albatross'
params.study_name = 'cmpt_trb_24'

# Main experimental settings
params.nP = 1000 # travellers
params.nV = 50 # drivers
params.nD = 3 # days
params.simTime = 8 # hours
params.wd_simulator = 'FleetPy'

# Other day-to-day settings
params.evol.drivers.kappa = 0.2
params.evol.drivers.res_wage.mean = 25 #euros/h
params.evol.drivers.gini = 0.35
params.evol.drivers.init_inc_ratio = 1 #expected income of informed drivers at start of sim as ratio of res wage
params.evol.drivers.inform.prob_start = 1 # probability of being informed at start of sim
params.evol.drivers.inform.beta = 0.1 # information transmission rate
params.evol.drivers.inform.std_fact = 0.5
params.evol.drivers.regist.prob_start = 0.2 # probability of being registered if informed at start of sim
params.evol.drivers.regist.beta = 0.2 # registration choice model parameter
params.evol.drivers.regist.cost_comp = 20 # daily share of registration costs (euros)
params.evol.drivers.regist.samp = 0.1 # probability of making (de)regist decision
params.evol.drivers.regist.min_work_exp = 0 # Working experience required before deregistration is possible
params.evol.drivers.regist.min_days = 5 # Minimum number of registered days before driver can deregister
params.evol.drivers.particip.beta = 0.1 # participation choice model parameter
params.evol.drivers.particip.probabilistic = True # stochasticity in participation choice
params.evol.travellers.inform.prob_start = 1
params.evol.travellers.inform.beta = 0.1
params.evol.travellers.inform.start_wait = 0
params.evol.travellers.inform.std_fact = 0.5
params.evol.travellers.reject_penalty = 30 * 60 # seconds
params.evol.travellers.kappa = 0.2
params.evol.travellers.min_prob = 0.05 # filtering criterion, when probability is lower when waiting time is zero, never consider RS
params.evol.travellers.mode_pref.mean_vot = 10 # Mean VoT in euro/h
params.evol.travellers.mode_pref.access_multip = 2
params.evol.travellers.mode_pref.wait_multip = 2.5
params.evol.travellers.mode_pref.bike_multip = 2
params.evol.travellers.mode_pref.beta_cost = -0.1592 # util/euro
params.evol.travellers.mode_pref.transfer_pen = 5 * 60 # seconds, to be added to IVT for each transfer
params.evol.travellers.mode_pref.ASC_car = -0.369 # util, rel to bike
params.evol.travellers.mode_pref.ASC_rs = -2.2497
params.evol.travellers.mode_pref.ASC_pt =  -1.5026
params.evol.travellers.mode_pref.ASC_car_sd = 0  # = 3.07
params.evol.travellers.mode_pref.ASC_rs_sd = 0  # 1.95
params.evol.travellers.mode_pref.ASC_pt_sd = 0  # 1.31
params.evol.travellers.mode_pref.ASC_bike_sd = 0  # 2.35
params.evol.travellers.mode_pref.gini = params.evol.drivers.gini

# Financial settings
params.platforms.base_fare = 1.5 #euro  #1.4
params.platforms.fare = 1.5 #euro/km #1.63
params.platforms.min_fare = 0 # euro
params.platforms.comm_rate = 0.25 #rate
params.drivers.fuel_costs = 0.25 #euro/km

# Properties alternative modes
params.alt_modes.pt.base_fare = 0.99 # euro
params.alt_modes.pt.km_fare = 0.174 # euro/km
params.alt_modes.car.km_cost = 0.5 # euro/km
params.alt_modes.car.diff_parking = True # different parking tariffs in city
params.alt_modes.car.park_cost = 7.5 # euro
params.alt_modes.car.park_cost_center = 15 # euro
params.alt_modes.car.access_time = 10 * 60 # s
params.speeds.bike = (1/2.5) * params.speeds.ride # m/s

# Demand settings
params.dist_threshold_min = 2000 # min dist
params.albatross = True  # if False, demand is artificially generated

# Regulation settings
params.platforms.reg_cap = np.inf
params.platforms.ptcp_cap = np.inf

# Start time
params.t0 = pd.Timestamp(2021, 11, 1, 9)

# Parameter type
# params.decis_type = "drivers"

def sample_space():
    # analysis of behavioural parameters
    space = DotMap()
    space.comm_rate = [0.25]
    space.repl_id = [0]
    return space


simulate_parallel(params=params, search_space=sample_space())
