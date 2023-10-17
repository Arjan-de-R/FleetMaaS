################################################################################
# Module: runners.py
# Description: Wrappers to prepare and run simulations
# Rafal Kucharski @ TU Delft
################################################################################

import os.path
import sys

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
from MaaSSim.src_MaaSSim.shared import prep_shared_rides
from MaaSSim.src_MaaSSim.utils import get_config, load_G, generate_demand, generate_vehicles, initialize_df, empty_series, \
    slice_space, read_vehicle_positions
from scipy.optimize import brute
import logging
import re
from MaaSSim.src_MaaSSim.d2d_sim import *
from MaaSSim.src_MaaSSim.d2d_demand import *
from MaaSSim.src_MaaSSim.d2d_supply import *
from MaaSSim.src_MaaSSim.decisions import dummy_False
from source.d2d.reproduce_MS_simulator import repl_sim_object
import zipfile
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

    scn_name = ''
    if params.platforms.service_types:
        cmpt_type_string = "".join([item[0] for item in params.platforms.service_types])
        scn_name = '-{}'.format(cmpt_type_string)
    for key, value in stamp.items():
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

    if len(inData.G) == 0:  # only if no graph in input
        inData = load_G(inData, params, stats=True, set_t=False)  # download graph for the 'params.city' and calc the skim matrices
        if params.alt_modes.car.diff_parking:
            inData = diff_parking(inData)  # determine which nodes are in center

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

    if inData is None:  # otherwise we use what is passed
        from MaaSSim.src_MaaSSim.data_structures import structures
        inData = structures.copy()  # fresh data
    if params is None:
            params = get_config(config, root_path = kwargs.get('root_path'))  # load from .json file
    if kwargs.get('make_main_path',False):
        from MaaSSim.src_MaaSSim.utils import make_config_paths
        params = make_config_paths(params, main = kwargs.get('make_main_path',False), rel = True)

    if params.paths.get('requests', False):
        inData = read_requests_csv(inData, params.paths.requests) # read request file
        if params.nP > inData.requests.shape[0]:
            raise Exception("Number of travellers is larger than demand dataset")
        if params.paths.get('PT_trips',False):
            pt_trips = load_OTP_result(params) # load output of OpenTripPlanner queries for the preprocessed requests and add resulting PT attributes to inData.requests
            inData.requests = pd.concat([inData.requests, pt_trips], axis=1)
            del pt_trips
        inData = sample_from_database(inData, params)  # sample nP and create inData.passengers
    else:
        # Generate requests - either based on a distribution or taken from Albatross - and corresponding passenger data
        inData = generate_demand(inData, params, avg_speed = True)

    if params.paths.get('vehicles', False):
        inData = read_vehicle_positions(inData, path=params.paths.vehicles)

    if len(inData.G) == 0:  # only if no graph in input
        inData = load_G(inData, params, stats=True)  # download graph for the 'params.city' and calc the skim matrices

    # Set random seeds used throughout the simulation
    np.random.seed(params.repl_id)
    random.seed(params.repl_id)

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
    inData.passengers = prefs_travs(inData, params)

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
    
    # Generate pool of job seekers, incl. setting multi-homing behaviour
    fixed_supply = generate_vehicles_d2d(inData, params)
    
    # correct 
    inData.vehicles = fixed_supply.copy()
    
    # Load path
    if path is None:
        path = os.getcwd()

    # Prepare schedule for shared rides and the within-day simulator
    inData = prep_shared_rides(inData, params.shareability)  # prepare schedules
    if params.paths.get('fleetpy_config', False): # initialise FleetPy
        fleetpy_dir = os.path.join(path, 'FleetPy')
        fleetpy_study_name = params.get('study_name', 'MaaSSim_FleetPy')
        config_file = params.paths.fleetpy_config
        network_name = params.city.split(",")[0]
        demand_name = network_name
        if not os.path.exists(os.path.join(fleetpy_dir, "data", "networks", network_name)):
            graphml_file = params.paths.G
            create_network_from_graphml(graphml_file, network_name)
        # Determine which MaaSSim nodes are in which zones
        zone_name = params.city.split(",")[0]
        inData.nodes['geometry'] = geopandas.points_from_xy(inData.nodes['x'],inData.nodes['y'])
        zones = geopandas.read_file(os.path.join(fleetpy_dir, "data", "zones", zone_name, "polygon_definition.geojson"))
        inData.nodes["zone_id"] = inData.nodes.apply(lambda row: get_init_zone_id(row, zones), axis=1)
        constant_config_file = os.path.join(fleetpy_dir,'studies','{}'.format(fleetpy_study_name),'scenarios','{}'.format(config_file))
        # Add zone id to passenger df
        inData.passengers['zone_id'] = inData.passengers.apply(lambda x: inData.nodes.zone_id.loc[x.pos], axis=1)
        # Expected (perceived) demand per zone (for the first day)
        perc_demand = pd.DataFrame(index=zones.zone_id, columns=['requests'])
        perc_demand.requests = 1 # assume equal demand in all zones for first day
        # Initialize MaaSSim simulator object to which FleetPy results are returned
        sim = repl_sim_object(inData, params=params, **kwargs)  
    else: # initialise MaaSSim within-day simulator
        sim = Simulator(inData, params=params,
                    kpi_veh = D2D_veh_exp,
                    kpi_pax = d2d_kpi_pax,
                    f_driver_out = D2D_driver_out,
                    f_trav_out = d2d_no_request,
                    f_trav_mode = dummy_False, **kwargs)  # initialize

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
    
    df_req = inData.requests[['pax_id','origin','destination','treq','dist','ttrav']]
    if 'haver_dist' in inData.requests.columns:
        df_req['haver_dist'] = inData.requests['haver_dist']
    df_pax = inData.passengers[['VoT','ASC_rs','ASC_pool','U_car','U_pt','U_bike', 'mode_without_rs', 'multihoming']]
    pd.concat([df_req, df_pax], axis=1).to_csv(os.path.join(result_path,'1_pax-properties.csv'))
    inData.vehicles[['pos', 'res_wage', 'multihoming']].to_csv(os.path.join(result_path,'2_driver-properties.csv'))
    inData.platforms.to_csv(os.path.join(result_path, '3_platform-properties.csv'))
    all_pax_df = pd.concat([all_req, all_pax], axis=1)
    all_pax_df = all_pax_df[all_pax_df.mode_choice != 'day-to-day']
    all_pax_df[['origin','destination','treq','dist','ttrav','VoT','ASC_rs','ASC_pool','U_car','U_pt','U_bike', 'mode_choice']].to_csv(os.path.join(result_path,'4_out-filter-pax.csv'))
    del all_pax, all_req, all_pax_df

    # Initialise convergence
    d2d_conv = pd.DataFrame()

    # Day-to-day simulator
    for day in range(params.get('nD', 1)):  # run iterations

        #----- Pre-day -----#
        inData.passengers = mode_preday(inData, params) # mode choice

        #----- Within-day simulator -----#
        if not params.paths.get('fleetpy_config', False): # run MaaSSim
            sim.make_and_run(run_id=day)  # prepare and SIM
            sim.output()  # calc results
            sim.last_res = sim.res[day].copy() # create a copy of the results - saved later
            del sim.res[day]
        else: # run FleetPy
            # Pre-day work choice
            inData.vehicles = work_preday(inData.vehicles, params)

            # Determine which platform(s) agents can use - using right FleetPy coding
            df_veh = inData.vehicles.copy()
            df_veh['ptcp_plf_index'] = df_veh.apply(lambda row: row.ptcp.nonzero()[0], axis=1)
            df_veh['ptcp_plf_index_string'] = df_veh.apply(lambda row: ';'.join(str(plf) for plf in np.nditer(row.ptcp_plf_index, flags=['zerosize_ok'])), axis=1)
            inData.vehicles.platform = df_veh['ptcp_plf_index_string']
            df_pax = inData.passengers.copy()
            df_pax['chosen_plf_index'] = df_pax.apply(lambda row: np.where((row.mode_day == 'rs') * row.registered)[0], axis=1)
            df_pax['chosen_plf_index_string'] = df_pax.apply(lambda row: ';'.join(str(plf) for plf in np.nditer(row.chosen_plf_index, flags=['zerosize_ok'])), axis=1)
            inData.passengers.platforms = df_pax.chosen_plf_index_string

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

            # FleetPy init: conversion from MaaSSim data structure
            fp_run_id = scn_name + '-day-{}'.format(day) # id in FleetPy
            transform_dtd_output_to_wd_input(dtd_result_dir, fleetpy_dir, fleetpy_study_name, network_name, fp_run_id, demand_name, params, zone_system_name=zone_name, exp_zone_demand=perc_demand)

            # Run FleetPy model
            scn_file = os.path.join(fleetpy_dir, "studies", fleetpy_study_name, "scenarios", f"{fp_run_id}.csv")
            run_scenarios(constant_config_file, scn_file)

            # FleetPy results: convert back to MaaSSim structure (simulator object) #TODO: o.a. indicators per platform, expected in-vehicle time, multi-homing vs single-homing
            sim = transform_wd_output_to_d2d_input(sim, fleetpy_dir, fleetpy_study_name, fp_run_id, inData)
            perc_demand = learn_demand(inData, params, zones, perc_demand)

        #----- Post-day -----#
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

        # Demand-side diffusion of platform information
        res_inf_trav = wom_trav(inData, travs_summary, params=params)
        inData.passengers.informed = res_inf_trav.informed
        inData = platform_regist_trav(inData, travs_summary, params=params)

        # Store KPIs of day
        dem_df, sup_df = d2d_summary_day(drivers_summary, travs_summary)
        dem_df.to_csv(os.path.join(result_path,'day_{}_travs.csv'.format(day)))
        sup_df.to_csv(os.path.join(result_path,'day_{}_drivers.csv'.format(day)))

        ### Determine convergence
        # Create new dataframe containing (experienced) participation values of all days
        travs_summary['requests_mh'] = travs_summary.apply(lambda row: row.requests.sum() > 1, axis=1)
        travs_summary['requests_sh_0'] = travs_summary.apply(lambda row: int(row['requests'][0]) * int(not row['requests_mh']), axis=1)
        drivers_summary['ptcp_mh'] = drivers_summary.apply(lambda row: ((~row.out).sum() > 1), axis=1)
        drivers_summary['ptcp_sh_0'] = drivers_summary.apply(lambda row: int(not row['out'][0]) * int(not row['ptcp_mh']), axis=1)
        conv_indic = pd.DataFrame([{'ptcp_dem_mh': travs_summary['requests_mh'].sum(), 'ptcp_dem_sh_0': travs_summary['requests_sh_0'].sum(), 
                                    'ptcp_sup_mh': drivers_summary['ptcp_mh'].sum(), 'ptcp_sup_sh_0': drivers_summary['ptcp_sh_0'].sum()}])
        if inData.platforms.shape[0] > 1: # more than one platform
            conv_indic['ptcp_dem_sh_1'] = travs_summary.apply(lambda row: int(row['requests'][1]) * int(not row['requests_mh']), axis=1).sum()
            conv_indic['ptcp_sup_sh_1'] = drivers_summary.apply(lambda row: int(not row['out'][1]) * int(not row['ptcp_mh']), axis=1).sum()
        d2d_conv = pd.concat([d2d_conv, conv_indic])
        # Create a copy of the csv by adding the last row to the already existing csv
        if day == 0: # include the headers on the first day
            conv_indic.to_csv(os.path.join(result_path,'5_conv-indicators.csv'), mode='a', index=False, header=True)
        else:
            conv_indic.to_csv(os.path.join(result_path,'5_conv-indicators.csv'), mode='a', index=False, header=False)
        
        if d2d_conv.shape[0] >= (params.convergence.get('first_moving_avg', 20) + params.convergence.get('second_moving_avg', 20) + params.convergence.get('req_steady_days', 10) + 1): # first day that convergence is possible
            if params.convergence.get('abs_ptcp_diff_dem', False) and params.convergence.get('abs_ptcp_diff_sup', False):
                rel_diff_ma_df = d2d_conv.rolling(params.convergence.get('first_moving_avg', 20)).mean().rolling(params.convergence.get('second_moving_avg', 20)).mean().tail(params.convergence.get('req_steady_days', 10) + 1).diff()
                rel_diff_ma_df['dem_mh_conv'] = rel_diff_ma_df.ptcp_dem_mh.abs() < params.convergence.abs_ptcp_diff_dem
                rel_diff_ma_df['dem_sh_0_conv'] = rel_diff_ma_df.ptcp_dem_sh_0.abs() < params.convergence.abs_ptcp_diff_dem
                rel_diff_ma_df['sup_mh_conv'] = rel_diff_ma_df.ptcp_sup_mh.abs() < params.convergence.abs_ptcp_diff_sup
                rel_diff_ma_df['sup_sh_0_conv'] = rel_diff_ma_df.ptcp_sup_sh_0.abs() < params.convergence.abs_ptcp_diff_sup
                if inData.platforms.shape[0] > 1:
                    rel_diff_ma_df['dem_sh_1_conv'] = rel_diff_ma_df.ptcp_dem_sh_1.abs() < params.convergence.abs_ptcp_diff_dem
                    rel_diff_ma_df['sup_sh_1_conv'] = rel_diff_ma_df.ptcp_sup_sh_1.abs() < params.convergence.abs_ptcp_diff_sup
                    conv_per_indicator = rel_diff_ma_df[['dem_mh_conv','dem_sh_0_conv','dem_sh_1_conv','sup_mh_conv','sup_sh_0_conv','sup_sh_1_conv']].all()
                else:
                    conv_per_indicator = rel_diff_ma_df[['dem_mh_conv','dem_sh_0_conv','sup_mh_conv','sup_sh_0_conv']].all()
            else:
                conv_factor = params.convergence.get('factor', 0.01)
                rel_diff_ma_df = d2d_conv.rolling(params.convergence.moving_avg).mean().tail(params.convergence.req_steady_days + 1).pct_change().tail(params.convergence.req_steady_days)
                conv_per_indicator = (rel_diff_ma_df.abs() < conv_factor).all()
            if conv_per_indicator.all():
                print('Scenario {} - All indicators have converged at end of day {}, day-to-day simulation is terminated.'.format(scn_name, day))
                break
            else:
                print('Scenario {} - Not all indicators have converged at end of day {}, next day is initialised.'.format(scn_name, day))
        else:
            print('Scenario {} - Initialisation period, simulation can not yet converge at end of day {}, next day is initialised.'.format(scn_name, day))
        
        del drivers_summary, travs_summary, dem_df, sup_df

    return sim


if __name__ == "__main__":
    # simulate(make_main_path='..')  # single run
    simulate()  # single run

    from MaaSSim.src_MaaSSim.utils import test_space

    simulate_parallel(search_space = test_space())
