################################################################################
# Module: runners.py
# Description: Wrappers to prepare and run simulations
# Rafal Kucharski @ TU Delft
################################################################################

import os.path
import sys
import time

from source.conversion.create_network_from_graphml import *
from source.conversion.trafo_FleetPy_to_MaaSSim import *
from source.conversion.trafo_MaaSSim_to_FleetPy import *
from source.d2d.supply import *
from source.d2d.demand import *
from source.d2d.platform import *

MAIN_DIR = os.path.dirname(__file__)
ABS_MAIN_DIR = os.path.abspath(MAIN_DIR)
FLEETPY_DIR = os.path.join(ABS_MAIN_DIR, "FleetPy")
MAASSIM_DIR = os.path.join(ABS_MAIN_DIR, "MaaSSim")
sys.path.append(ABS_MAIN_DIR)
sys.path.append(FLEETPY_DIR)
# sys.path.append(MAASSIM_DIR)

from MaaSSim.src_MaaSSim.maassim import Simulator
from MaaSSim.src_MaaSSim.utils import get_config, load_G, generate_demand, generate_vehicles, initialize_df, empty_series, \
    slice_space, read_vehicle_positions, create_seconds_of_day
from scipy.optimize import brute
import logging
import re
from MaaSSim.src_MaaSSim.d2d_sim import *
from MaaSSim.src_MaaSSim.d2d_demand import *
from MaaSSim.src_MaaSSim.d2d_supply import *
from MaaSSim.src_MaaSSim.decisions import dummy_False
from source.d2d.reproduce_MS_simulator import repl_sim_object
from tmc.utils import *
import json
import geopandas
from FleetPy.run_examples import run_scenarios

def single_pararun(one_slice, *args):
    # function to be used with optimize brute
    inData, params, search_space = args  # read static input
    _inData = inData.copy()
    _params = params.copy()
    stamp = dict()
    # parameterize
    for i, key in enumerate(search_space.keys()):
        val = search_space[key][int(one_slice[int(i)])]
        stamp[key] = val
        _params = return_scn_params(params, key, val)

    scn_name = '{}-'.format(params.get('dem_mgmt', 'None'))
    if params.platforms.service_types:
        cmpt_type_string = "".join([item[0] for item in params.platforms.service_types])
        dem_mgmt_string = params.get('dem_mgmt', 'None')
        if dem_mgmt_string == 'tmc':
            if params.get('charging_scheme'):
                dem_mgmt_string = 'tmc{}_{}'.format(params.tmc.allocated_credits_per_day, params.get('charging_scheme'))
            else:
                dem_mgmt_string = 'tmc{}'.format(params.tmc.allocated_credits_per_day)
        elif dem_mgmt_string == 'cgp':
            if params.get('city_charge'):
                dem_mgmt_string = 'city_cgp{}'.format(params.city_charge.car)
            else:
                dem_mgmt_string = 'cgp{}'.format(params.zone_charge.car)
        scn_name = '-{}-{}'.format(dem_mgmt_string, cmpt_type_string)
    for key, value in stamp.items():
        if key != 'service_types':
            scn_name += '-{}-{}'.format(key, value)
    scn_name = re.sub('[^-a-zA-Z0-9_.() ]+', '', scn_name)[1:]

    # Set-up simulation log
    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists(os.path.join('results',scn_name)):
        os.mkdir(os.path.join('results',scn_name))
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename=os.path.join('results','{}'.format(scn_name), '00_simulation.log'),  # File to save logs
                    filemode='a')       # Append mode for the log file
    logger_d2d = logging.getLogger("logger_d2d")
    logger_d2d.setLevel(logging.INFO)

    handler = logging.FileHandler(os.path.join('results','{}'.format(scn_name), '00_simulation.log'))
    handler.setLevel(logging.INFO)
    logger_d2d.addHandler(handler)

    sim = simulate(inData=_inData, params=_params, logger_level=logging.INFO, scn_name = scn_name)

    print(scn_name, pd.Timestamp.now(), 'end')
    return 0


def simulate_parallel(config="MaaSSim/data/config/parallel.json", inData=None, params=None, search_space=None, **kwargs):
    if inData is None:  # otherwise we use what is passed
        from MaaSSim.src_MaaSSim.data_structures import structures
        inData = structures.copy()  # fresh data
    if params is None:
        params = get_config(config, root_path = kwargs.get('root_path'))  # load from .json file

    brute(func=single_pararun,
          ranges=slice_space(search_space, replications=params.parallel.get("nReplications",1)),
          args=(inData, params, search_space),
          full_output=True,
          finish=None,
          workers=params.parallel.get('nThread',1))


def simulate(config="data/config.json", inData=None, params=None, path = None, **kwargs):
    """
    main runner and wrapper
    loads or uses json config to prepare the data for simulation, run it and process the results
    :param config: .json file path
    :param inData: optional input data
    :param params: loaded json file
    :param kwargs: optional arguments
    :return: simulation object with results
    """

    # Load path
    if path is None:
        path = os.getcwd()

    if inData is None:  # otherwise we use what is passed
        from MaaSSim.src_MaaSSim.data_structures import structures
        inData = structures.copy()  # fresh data
    if params is None:
        params = get_config(config, root_path = kwargs.get('root_path'))  # load from .json file
    if kwargs.get('make_main_path',False):
        from MaaSSim.src_MaaSSim.utils import make_config_paths
        params = make_config_paths(params, main = kwargs.get('make_main_path',False), rel = True)

    # Set random seeds used for setting up simulation (not for the simulation itself)
    np.random.seed(0)
    random.seed(0)

    if params.paths.get('requests', False):
        inData = read_requests_csv(inData, params) # read request file
        if params.speeds.get('bike', False):
            inData.requests['ttrav_bike'] = inData.requests['dist_bike'] / params.speeds.bike + params.alt_modes.bike.get('access_time', False) # Overwrite bike travel time
        if (params.dem_mgmt == 'cgp') and 'through_center' not in inData.requests.keys():
            # Determine which trips pass through city centre, based on shortest path (only if not yet preprocessed)
            nw, osm_to_fp_ids = prep_fp_shortest_paths(params)
            centre_nodes = nodes_in_centre(params)
            inData.requests['through_center'] = inData.requests.apply(lambda row: shortest_path_through_area(nw, osm_to_fp_ids, centre_nodes, row.origin, row.destination), axis=1)
            inData.requests['pax_id'] = inData.requests.index
            inData.requests.to_csv(os.path.join(path, 'source', 'MaaSSim', 'data', 'demand', 'reqs_center.csv'))
        inData.requests.index = pd.RangeIndex(start=0, stop=len(inData.requests), name=inData.requests.index.name)
        inData.passengers['orig_person_id'] = inData.passengers.index
        inData.passengers.index = inData.requests.index
        inData.requests.index.name = 'pax_id'
        inData.passengers.index.name = 'pax_id'
        params.nP = inData.requests.shape[0]
    else:
        # Generate requests - either based on a distribution or taken from Albatross - and corresponding passenger data
        inData = generate_demand(inData, params, avg_speed = False)

    if params.paths.get('vehicles', False):
        inData = read_vehicle_positions(inData, path=params.paths.vehicles)

    nw_type = params.get("network_type", "MaaSSim")
    if len(inData.G) == 0 and nw_type == "MaaSSim":  # only if no graph in input and network type is MaaSSim
        inData = load_G(inData, params, stats=True)  # download graph for the 'params.city' and calc the skim matrices
    if params.alt_modes.car.get('diff_parking', False):
        inData, centre_nodes = prep_inData_nodes_centre(inData, params)  # determine which nodes are in center
    if 'through_center' not in inData.requests.columns:
        inData.requests['through_center'] = False

    # Set properties of platform(s)
    inData.platforms = pd.concat([inData.platforms,pd.DataFrame(columns=['base_fare','comm_rate','min_fare','match_obj','max_wait_time','max_rel_detour'])])
    inData.platforms = initialize_df(inData.platforms)
    if not params.platforms.get('service_types'): # if service type(s) are not provided
        params.platforms.service_types = ['solo']
    for plat_id in range(0,len(params.platforms.service_types)):
        # initialise solo platform
        if params.platforms.service_types[plat_id] == 'solo':
            inData.platforms.loc[plat_id] = init_solo_plf(params, plat_id)
        else:
            inData.platforms.loc[plat_id] = init_pooling_plf(params, plat_id)

    # Generate mode preferences
    if params.dem_mgmt:
        if 'schedule_id' in inData.requests.columns:
            inData.requests = inData.requests.drop(['schedule_id'], axis=1)
        inData.passengers = prefs_travs_tmc(inData, params)
    else:
        inData.passengers = prefs_travs(inData, params)

    # Determine required mobility credits per mode for each trip request, as well as trading perceptions
    if params.dem_mgmt == 'tmc':
        inData.requests = trip_credit_cost(inData, params)
        inData.passengers = set_trading_prefs(inData.passengers, params)

    if not params.get('dem_mgmt'):
        all_req = inData.requests.copy()
        all_pax = mode_filter(inData, params)
        inData.passengers = all_pax[all_pax.mode_choice == "day-to-day"]
        inData.requests = inData.requests[inData.requests.index.isin(inData.passengers.index)]
        inData.passengers.reset_index(drop=True, inplace=True)
    inData.requests.reset_index(drop=True, inplace=True)
    inData.requests['pax_id'] = inData.requests.index

    # Generate information available to travellers at the start of the simulation, and whether travellers are willing to multi-home
    inData.passengers = set_multihoming_travellers(inData.passengers, params)
    inData.passengers['informed'] = np.random.rand(len(inData.passengers)) < params.evol.travellers.inform.prob_start
    inData.passengers = start_regist_travs(inData, params)

    if params.dem_mgmt == 'tmc':
        # Set starting mobility credit balance
        credits_per_day = params.tmc.get('allocated_credits_per_day', 10)
        inData.passengers['tmc_balance'] = determine_starting_balance(inData, params, credits_per_day)
        inData.passengers['money_balance'] = 0
        inData.passengers['tot_credit_bought'] = 0
        inData.passengers['tot_credit_sold'] = 0
        # Establish possible buy/sell actions, i.e. what are possible credit prices, balance values and buy quantities (if specified)
        buy_table_dims = buy_table_dimensions(params)
    
    # Prepare schedule for the within-day simulator
    fleetpy_dir = os.path.join(path, 'FleetPy')
    fleetpy_study_name = params.get('study_name', 'MaaSSim_FleetPy')
    config_file = params.paths.fleetpy_config
    constant_config_file = os.path.join(fleetpy_dir,'studies','{}'.format(fleetpy_study_name),'scenarios','{}'.format(config_file))
    network_name = params.city.split(",")[0]
    demand_name = network_name
    if not os.path.exists(os.path.join(fleetpy_dir, "data", "networks", network_name)):
        graphml_file = params.paths.G
        create_network_from_graphml(graphml_file, network_name, params)
    bike_network_name = '{}_bike'.format(network_name)
    if not os.path.exists(os.path.join(fleetpy_dir, "data", "networks", bike_network_name)):
        graphml_file = params.paths.G
        create_network_from_graphml(graphml_file, bike_network_name, params)
    # Determine which MaaSSim nodes are in which zones
    zone_name = params.city.split(",")[0]
    zones = geopandas.read_file(os.path.join(fleetpy_dir, "data", "zones", zone_name, "polygon_definition.geojson"))
    if os.path.isfile(os.path.join(fleetpy_dir, "data", "zones", zone_name, network_name, "node_zone_info.csv")):
        node_zone_df = pd.read_csv(os.path.join(fleetpy_dir, "data", "zones", zone_name, network_name, "node_zone_info.csv"), index_col="Unnamed: 0")
         # Add zone id to passenger df
        inData.passengers['zone_id'] = inData.requests.apply(lambda x: node_zone_df.loc[x.origin]['zone_id'], axis=1)
    else:
        inData.nodes['geometry'] = geopandas.points_from_xy(inData.nodes['x'],inData.nodes['y'])
        inData.nodes["zone_id"] = inData.nodes.apply(lambda row: get_init_zone_id(row, zones), axis=1)
        # Add zone id to passenger df
        inData.passengers['zone_id'] = inData.passengers.apply(lambda x: inData.nodes.zone_id.loc[x.pos], axis=1)

    # Load ride-hailing zones
    rh_zone_f = os.path.join(fleetpy_dir, "data", "zones", zone_name, network_name, "rh_zones.csv")
    municipality_zone_f = os.path.join(fleetpy_dir, "data", "zones", zone_name, "general_information.csv")
    if os.path.isfile(rh_zone_f) and os.path.isfile(municipality_zone_f):
        rh_zones = pd.read_csv(rh_zone_f, index_col=False)
        df_municipality_zone = pd.read_csv(municipality_zone_f, index_col=False)
        # Step 1: Merge zone_id → municipality into df_travellers
        inData.passengers = inData.passengers.merge(df_municipality_zone[['zone_id', 'municipality']], on='zone_id', how='left')
        # Step 2: Merge municipality → rh_zone
        inData.passengers = inData.passengers.merge(
            rh_zones, on='municipality', how='left'
        )
        inData.passengers.drop(columns=['municipality'], inplace=True)
        rh_zone_list = inData.passengers['rh_zone'].unique()

    # # Expected share of (perceived) demand per zone (for the first day) -- we assume that they consider (and know) all travel demand in the network
    all_zone_ids = zones['zone_id'].unique()
    perc_demand = inData.passengers['zone_id'].value_counts().reindex(all_zone_ids, fill_value=0).sort_index() * (1/inData.passengers.shape[0])
    perc_demand.name = "requests"
    perc_demand = perc_demand.to_frame()

    # Generate pool of job seekers, incl. setting multi-homing behaviour
    if len(inData.nodes) == 0:
        inData.nodes = node_zone_df['node_index'].copy()
        inData.nodes = inData.nodes.rename('name')

    # Load vehicles
    fixed_supply = generate_vehicles_d2d(inData, params)
    inData.vehicles = fixed_supply.copy()

    # Initialize MaaSSim simulator object to which FleetPy results are returned
    sim = repl_sim_object(inData, params=params, **kwargs)  

    # Where are the (final) results of the day-to-day simulation be stored
    scn_name = kwargs.get('scn_name')
    if not os.path.exists(os.path.join(path,'results')):
        os.mkdir(os.path.join(path,'results'))
    result_path = os.path.join(path, 'results', scn_name)
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    elif os.path.exists(os.path.join(result_path,"5_perc-utilities.csv")):
        os.remove(os.path.join(result_path,"5_perc-utilities.csv"))
    
    params.t0 = params.t0.to_pydatetime().strftime('%Y-%m-%d %H:%M:%S')
    with open(os.path.join(result_path, '0_params.json'), 'w') as json_file:
        json.dump(params, json_file)
    
    df_req = inData.requests[['pax_id','origin','destination','treq','dist','ttrav', 'ttrav_bike', 'transitTime', 'waitingTime', 'PTfare', 'through_center']]
    df_req = df_req.rename(columns={'ttrav':'ttrav_car', 'waitingTime': 'PT_waitingTime', 'transitTime': 'PT_ivTime'})
    if 'haver_dist' in inData.requests.columns:
        df_req.loc[:, 'haver_dist'] = inData.requests['haver_dist']
    df_pax_cols = [x for x in ['VoT','ASC_rs','ASC_pool','ASC_car','ASC_pt','ASC_bike','U_car','U_pt','U_bike', 'mode_without_rs', 'multihoming','bike_option', 'car_option', 'zone_id', 'rh_zone'] if x in inData.passengers.columns]
    df_pax = inData.passengers[df_pax_cols]
    pd.concat([df_req, df_pax], axis=1).to_csv(os.path.join(result_path,'1_pax-properties.csv'))
    inData.vehicles[['pos', 'res_wage', 'multihoming']].to_csv(os.path.join(result_path,'2_driver-properties.csv'))
    inData.platforms.to_csv(os.path.join(result_path, '3_platform-properties.csv'))
    if not params.get('dem_mgmt'):
        all_pax_df = pd.concat([all_req, all_pax], axis=1)
        all_pax_df = all_pax_df[all_pax_df.mode_choice != 'day-to-day']
        all_pax_df[['origin','destination','treq','dist','ttrav','VoT','ASC_rs','ASC_pool','U_car','U_pt','U_bike', 'mode_choice']].to_csv(os.path.join(result_path,'4_out-filter-pax.csv'))
        del all_pax, all_req, all_pax_df

    # Initialise credit price
    credit_price = None

    # Starting congestion levels
    params.t0 = create_seconds_of_day(params.t0)
    if params.paths.get('ttfs', False):
        inData.tt_factors = pd.read_csv(params.paths.ttfs, index_col=False)
        inData.tt_factors['simulation_time'] = inData.tt_factors['simulation_time'] + params.t0
        inData.tt_factors.set_index('simulation_time', inplace=True)
        inData.tt_factors['travel_time_factor'] = inData.tt_factors['travel_time_factor'].astype(float)
    else:
        ttf_update_interval = int(params.congestion.get('update_interval', params.simTime) * 3600)
        inData.tt_factors = pd.DataFrame(index=range(params.t0, params.t0 + params.simTime*3600, ttf_update_interval), columns=['travel_time_factor'])
        inData.tt_factors['travel_time_factor'] = params.congestion.get('start_ttf', 1)
        inData.tt_factors['background_traffic_share'] = 1 / inData.tt_factors.shape[0]
        inData.tt_factors.index.name = 'simulation_time'

    # Update expected travel times for car and ride-hailing based on congestion levels
    inData.passengers['expected_ttf'] = inData.requests.apply(lambda row: compute_weighted_avg_ttf(row['treq'], row['ttrav'], inData.tt_factors, params), axis=1)

    # Initialise license plate rationing
    if params.dem_mgmt == 'lpr':
        inData.passengers['odd_license'] = (np.random.randint(2, size = inData.passengers.shape[0]) == 1)

    # Initialise convergence
    d2d_conv = pd.DataFrame()
    conv_dict = dict()
    conv_dict['expected_modal_split'] = pd.DataFrame()
    mode_choice_converged = False
    mode_choice_iter = 0

    # Set random seeds used throughout the simulation
    np.random.seed(params.repl_id)
    random.seed(params.repl_id)

    # Simulator
    while mode_choice_iter < params.convergence.get('max_iter_mode_choice', 100):
        print(f"--- Mode Choice Iteration {mode_choice_iter} ---")
        transport_model_converged = False
        transport_iter = 0
    
        #----- Pre-day -----#
        section_start = time.time()

        # Credit trading
        if params.dem_mgmt == 'tmc':
            credit_price, market_orders, probabilities, gtt_chosen_mode, expected_attr = price_and_orders(inData, params, possible_prices=buy_table_dims['price'], prev_price=credit_price)
            # First, determine convergence (expected modal splits per time period)
            expected_modal_splits = expected_modal_split_per_time_period(inData, params, probabilities)
            conv_dict['expected_modal_split'] = store_expected_modal_split(expected_modal_splits, mode_choice_iter, conv_dict['expected_modal_split'])
            # Next, we have to check for each time period if all market shares have not changed by more than x% (moving average)
            mode_choice_converged = determine_mode_choice_convergence(conv_dict['expected_modal_split'], params)
            if mode_choice_converged:
                break
            inData.passengers['mode_day'] = probabilities['decis']
            inData.requests['chosen_mode_perc_gtt'] = gtt_chosen_mode
            print(f"Section 1 -- determination market price and mode choice -- completed in {time.time() - section_start:.4f} seconds")
            section_start = time.time()
            satisfied_orders, denied_orders = market_transactions(market_orders)
            print(f"Section 2 -- ordering and trading -- completed in {time.time() - section_start:.4f} seconds")
            section_start = time.time()
            # Update credit and monetary balance
            inData.passengers = update_balances(inData, satisfied_orders, denied_orders)
            # Save trading market indicators
            save_tmc_market_indicators(inData, result_path, mode_choice_iter, transport_iter, credit_price, satisfied_orders, denied_orders)
            print(f"Section 3 -- updating balance and saving market indicators -- completed in {time.time() - section_start:.4f} seconds")
            section_start = time.time()

        #----- Within-day simulator -----#
        # Pre-day work choice
        if not params.evol.drivers.particip.auto:
            inData.vehicles = work_preday(inData.vehicles, params)
        else:
            inData.vehicles = determine_centralised_rh_fleet(inData, params)

        # Determine which platform(s) agents can use - using right FleetPy coding
        df_veh = inData.vehicles.copy()
        df_veh['ptcp_plf_index'] = df_veh.apply(lambda row: row.ptcp.nonzero()[0], axis=1)
        df_veh['ptcp_plf_index_string'] = df_veh.apply(lambda row: ';'.join(str(plf) for plf in np.nditer(row.ptcp_plf_index, flags=['zerosize_ok'])), axis=1)
        inData.vehicles.platform = df_veh['ptcp_plf_index_string']
        df_pax = inData.passengers.copy()
        if params.evol.travellers.plf_choice == 'preday':
            df_pax['chosen_plf_index_string'] = df_pax.apply(lambda row: row.mode_day.split("_")[-1] + "" if row.mode_day.startswith('rs_') else "", axis=1)
        else:
            df_pax['chosen_plf_index'] = df_pax.apply(lambda row: np.where((row.mode_day == 'rs') * row.registered)[0], axis=1)
            df_pax['chosen_plf_index_string'] = df_pax.apply(lambda row: ';'.join(str(plf) for plf in np.nditer(row.chosen_plf_index, flags=['zerosize_ok'])), axis=1)
        inData.passengers['platforms'] = df_pax.chosen_plf_index_string

        # Generate input csv's for FleetPy
        dtd_result_dir = os.path.join(path, 'temp_res','{}'.format(scn_name))
        if not os.path.exists(dtd_result_dir):
            if not os.path.exists(os.path.join(path,'temp_res')):
                os.mkdir(os.path.join(path,'temp_res'))
            os.mkdir(dtd_result_dir)
        inData.requests.to_csv(os.path.join(dtd_result_dir,'inData_requests.csv'))
        inData.passengers.to_csv(os.path.join(dtd_result_dir,'inData_passengers.csv')) 
        inData.vehicles.to_csv(os.path.join(dtd_result_dir,'inData_vehicles.csv')) 
        inData.platforms.to_csv(os.path.join(dtd_result_dir,'inData_platforms.csv'))

        while not transport_model_converged and transport_iter < params.convergence.get('max_iter_transport', 20):
            print(f"- Transport model iteration {transport_iter} -")
            inData.tt_factors.to_csv(os.path.join(dtd_result_dir,'inData_ttfs.csv'))
            save_ttfs_to_d2d_csv(inData, result_path, mode_choice_iter, transport_iter)

            # FleetPy init: conversion from MaaSSim data structure
            fp_run_id = scn_name + '-mc-{}-tm-{}'.format(mode_choice_iter, transport_iter) # id in FleetPy
            transform_dtd_output_to_wd_input(dtd_result_dir, fleetpy_dir, fleetpy_study_name, network_name, nw_type, fp_run_id, demand_name, params, zone_system_name=zone_name, exp_zone_demand=perc_demand)
            print(f"Section 4 -- FleetPy initialisation -- completed in {time.time() - section_start:.4f} seconds")
            section_start = time.time()

            # Run FleetPy model
            scn_file = os.path.join(fleetpy_dir, "studies", fleetpy_study_name, "scenarios", f"{fp_run_id}.csv")
            run_scenarios(constant_config_file, scn_file)
            print(f"Section 5 -- running FleetPy -- completed in {time.time() - section_start:.4f} seconds")
            section_start = time.time()

            # FleetPy results: convert back to MaaSSim structure (simulator object) #TODO: o.a. indicators per platform, expected in-vehicle time, multi-homing vs single-homing
            sim = transform_wd_output_to_d2d_input(sim, fleetpy_dir, fleetpy_study_name, fp_run_id, inData)
            print(f"Section 6 -- converting FleetPy results -- completed in {time.time() - section_start:.4f} seconds")
            section_start = time.time()

            #----- Post-day -----#
            ## Determine new travel time factors (congestion levels)
            ttfs = determine_congestion(params, inData, network_name, fp_run_id, fleetpy_dir, fleetpy_study_name)
            inData.tt_factors['travel_time_factor'] = ttfs['new_travel_time_factor'].copy()
            # Determine whether congestion factors have sufficiently converged
            transport_model_converged = ttf_convergence_check(params, ttfs)
            if not transport_model_converged:
                transport_iter += 1
            print(f"Section 7 -- determine road congestion and convergence -- completed in {time.time() - section_start:.4f} seconds")

        if transport_model_converged:
            # Determine key KPIs
            drivers_summary = update_d2d_drivers(sim=sim, params=params)
            travs_summary = update_d2d_travellers(sim=sim, params=params, pax=inData.passengers)
            
            # Update work experience of job seekers
            exp_df = update_work_exp(inData, drivers_summary)   # number of days work experience
            inData.vehicles.work_exp = exp_df.work_exp

            # Supply-side diffusion of platform information
            inData.vehicles.informed = wom_driver(inData, params=params)   # which job seekers are informed about ride-hailing
            
            # (De-)registration decisions
            inData.vehicles = platform_regist_driver(inData, drivers_summary, params=params)
            inData.vehicles.pos = fixed_supply.pos

            # Determine congestion charge paid
            if params.dem_mgmt == 'cgp':
                inData.passengers['paid_cgp'] = determine_congestion_charge(inData, params)

            # Learning ride-hailing kpi's for travellers
            inData.passengers = learn_wd_kpis(inData, travs_summary, params, mode_choice_iter, transport_iter, result_path, rh_zone_list)
            perc_demand = learn_demand(inData, params, zones, perc_demand)

            section_start = time.time()

            # Store KPIs of iteration
            travs_summary, warm_pax_df, warm_ttf_df = filter_warm_period(travs_summary, inData, params) # TODO: also apply warm-up filter for drivers (needs to be done in FleetPy?)
            dem_df, sup_df = d2d_summary_day(inData, drivers_summary, travs_summary, warm_pax_df)
            dem_df.to_csv(os.path.join(result_path,'mc_{}_tm_{}_travs.csv'.format(mode_choice_iter, transport_iter)))
            sup_df.to_csv(os.path.join(result_path,'mc_{}_tm_{}_drivers.csv'.format(mode_choice_iter, transport_iter)))

            ### Determine and store day's key KPIs, and determine convergence
            fp_result_dir = os.path.join(fleetpy_dir, 'studies', fleetpy_study_name, 'results', fp_run_id) # where are the results stored
            congest_indic = determine_vkt(inData, params, fp_result_dir)
            d2d_conv = save_market_shares(inData, params, result_path, mode_choice_iter, travs_summary, drivers_summary, d2d_conv, congest_indic)
            save_random_states(result_path)

            del drivers_summary, travs_summary, dem_df, sup_df, warm_ttf_df, warm_pax_df
            print(f"Section 8 -- post-processing day -- completed in {time.time() - section_start:.4f} seconds")

        mode_choice_iter += 1    

    # Save expected traveller attributes and probabilities
    expected_attr_df = pd.DataFrame(expected_attr)
    expected_attr_df.to_csv(os.path.join(result_path,'conv_trav_expected_attr.csv'))
    probabilities.drop(columns=['decis', 'chosen_rs_plf']).to_csv(os.path.join(result_path,'conv_trav_mode_probabilities.csv'))

    # Final output
    if mode_choice_converged:
        print("Simulation converged successfully.")
        # Save convergence status in output folder
        with open(os.path.join(result_path, 'convergence_status.txt'), 'w') as f:
            f.write("Simulation converged successfully.")
    else:
        print("Simulation reached max iterations without full convergence.")
        with open(os.path.join(result_path, 'convergence_status.txt'), 'w') as f:
            f.write("Simulation reached max mode choice iterations without full convergence.")

    return sim


if __name__ == "__main__":
    # simulate(make_main_path='..')  # single run
    simulate()  # single run

    from MaaSSim.src_MaaSSim.utils import test_space

    simulate_parallel(search_space = test_space())
