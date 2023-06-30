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

MAIN_DIR = os.path.dirname(__file__)
ABS_MAIN_DIR = os.path.abspath(MAIN_DIR)
FLEETPY_DIR = os.path.join(ABS_MAIN_DIR, "FleetPy")
MAASSIM_DIR = os.path.join(ABS_MAIN_DIR, "MaaSSim")
sys.path.append(ABS_MAIN_DIR)
sys.path.append(FLEETPY_DIR)
# sys.path.append(MAASSIM_DIR)

from MaaSSim.MaaSSim.maassim import Simulator
from MaaSSim.MaaSSim.shared import prep_shared_rides
from MaaSSim.MaaSSim.utils import get_config, load_G, generate_demand, generate_vehicles, initialize_df, empty_series, \
    slice_space, read_requests_csv, read_vehicle_positions
from scipy.optimize import brute
import logging
import re
from MaaSSim.MaaSSim.d2d_sim import *
from MaaSSim.MaaSSim.d2d_demand import *
from MaaSSim.MaaSSim.d2d_supply import *
from MaaSSim.MaaSSim.decisions import dummy_False
from source.d2d.reproduce_MS_simulator import repl_sim_object

import zipfile
import json
from FleetPy.run_examples import run_scenarios

# import functions from FleetPy
# import functions from FleetMaaS - conversion scripts


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

        if key in ['comm_rate', 'fare', 'base_fare', 'reg_cap', 'ptcp_cap']:
            _params.platforms[key] = val
        if key == 'gini':
            _params.evol.drivers[key] = val
            _params.evol.travellers.mode_pref[key] = val
        else:
            _params[key] = val

    # stamp['dt'] = str(pd.Timestamp.now()).replace('-','').replace('.','').replace(' ','')

    scn_name = ''
    for key, value in stamp.items():
        scn_name += '-{}_{}'.format(key, value)
    scn_name = re.sub('[^-a-zA-Z0-9_.() ]+', '', scn_name)

    sim = simulate(inData=_inData, params=_params, logger_level=logging.WARNING, scn_name = scn_name)

    print(scn_name, pd.Timestamp.now(), 'end')
    return 0


def simulate_parallel(config="MaaSSim/data/config/parallel.json", inData=None, params=None, search_space=None, **kwargs):
    if inData is None:  # otherwise we use what is passed
        from MaaSSim.MaaSSim.data_structures import structures
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
        from MaaSSim.MaaSSim.data_structures import structures
        inData = structures.copy()  # fresh data
    if params is None:
            params = get_config(config, root_path = kwargs.get('root_path'))  # load from .json file
    if kwargs.get('make_main_path',False):
        from MaaSSim.MaaSSim.utils import make_config_paths
        params = make_config_paths(params, main = kwargs.get('make_main_path',False), rel = True)

    if params.paths.get('requests', False):
        inData = read_requests_csv(inData, path=params.paths.requests)

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
    inData.platforms.loc[0]=[params.platforms.fare,'Platform 0',30,params.platforms.base_fare,params.platforms.comm_rate,params.platforms.min_fare,params.platforms.match_obj,params.platforms.max_wait_time,params.platforms.max_rel_detour,]
    for plat_id in range(1,params.get('nS', 1)): # more than 1 platform
        inData.platforms.loc[plat_id]=[params.platforms.get('fare_{}'.format(plat_id),params.platforms.fare),'Platform {}'.format(plat_id),30,
                                    params.platforms.get('base_fare_{}'.format(plat_id),params.platforms.base_fare),
                                    params.platforms.get('comm_rate_{}'.format(plat_id), params.platforms.comm_rate),
                                    params.platforms.get('min_fare_{}'.format(plat_id),params.platforms.min_fare),
                                    params.platforms.get('match_obj_{}'.format(plat_id),params.platforms.match_obj),
                                    params.platforms.get('max_wait_time_{}'.format(plat_id),params.platforms.max_wait_time),
                                    params.platforms.get('max_rel_detour_{}'.format(plat_id),params.platforms.max_rel_detour)]

    # Generate requests - either based on a distribution or taken from Albatross - and corresponding passenger data
    if params.get('albatross', False):
        inData = load_albatross_proc(inData, params, avg_speed = True)
        inData.requests = inData.requests.drop(['orig_geo', 'dest_geo', 'origin_y', 'origin_x', 'destination_y', 'destination_x', 'time'], axis = 1)
        inData = sample_from_alba(inData, params)
    else:
        inData = generate_demand(inData, params, avg_speed = True)

    inData.passengers = prefs_travs(inData, params)

    # Load processed Albatross file, the OTP result, and compute PT fares
    # inData.pt_itinerary = load_OTP_result(params)

    # Determine whether to consider all generated requests in day-to-day simulation or only those that are relatively likely to consider ride-hailing
    if params.evol.travellers.get('min_prob', 0) > 0:
        all_pax = mode_filter(inData, params)
        inData.passengers = all_pax[all_pax.mode_choice == "day-to-day"]
        inData.requests = inData.requests[inData.requests.pax_id.isin(inData.passengers.index)]
        # inData.pt_itinerary = inData.pt_itinerary[inData.pt_itinerary.pax_id.isin(inData.passengers.index)] # TODO: check compatibility PT alternative
        inData.passengers.reset_index(drop=True, inplace=True)
        inData.requests.reset_index(drop=True, inplace=True)
        # inData.pt_itinerary.reset_index(drop=True, inplace=True)
        inData.requests['pax_id'] = inData.requests.index
        # inData.pt_itinerary['pax_id'] = inData.pt_itinerary.index
    else:
        all_pax = inData.passengers.copy()
        all_pax['mode_choice'] == "day-to-day"

    # Generate information available to travellers at the start of the simulation, and whether travellers are willing to multi-home
    inData.passengers['informed'] = np.random.rand(len(inData.passengers)) < params.evol.travellers.inform.prob_start
    inData.passengers['expected_wait'] = params.evol.travellers.inform.start_wait
    inData.passengers = start_regist_travs(inData.passengers, params)
    inData.passengers = set_multihoming_travellers(inData.passengers, params)
    
    # Generate pool of job seekers, incl. setting multi-homing behaviour
    fixed_supply = generate_vehicles_d2d(inData, params)
    fixed_supply = set_multihoming_drivers(fixed_supply, params)
    inData.vehicles = fixed_supply.copy()
    
    # Load path
    if path is None:
        path = os.getcwd()

    # Prepare schedule for shared rides and the within-day simulator
    inData = prep_shared_rides(inData, params.shareability)  # prepare schedules
    if params.get('wd_simulator', 'MaaSSim') == 'MaaSSim':
        sim = Simulator(inData, params=params,
                    kpi_veh = D2D_veh_exp,
                    kpi_pax = d2d_kpi_pax,
                    f_driver_out = D2D_driver_out,
                    f_trav_out = d2d_no_request,
                    f_trav_mode = dummy_False, **kwargs)  # initialize
    if params.get('wd_simulator', 'MaaSSim') == 'FleetPy':
        # Initialise FleetPy
        fleetpy_dir = os.path.join(path, 'FleetPy')
        fleetpy_study_name = params.get('study_name', 'MaaSSim_FleetPy')
        config_file = params.fleetpy_config
        network_name = params.city.split(",")[0]
        demand_name = network_name
        # graphml_file = params.paths.G
        # create_network_from_graphml(graphml_file, network_name) # used only for preprocessing
        constant_config_file = os.path.join(fleetpy_dir,'studies','{}'.format(fleetpy_study_name),'scenarios','{}'.format(config_file))
        sim = repl_sim_object(inData, params=params, **kwargs)  # initialize MaaSSim simulator object to which FleetPy results are returned

    # Where are the (final) results of the day-to-day simulation be stored
    scn_name = kwargs.get('scn_name')
    sim_zip = zipfile.ZipFile(os.path.join(path, 'results', '{}.zip'.format(scn_name)), 'w')
    params.t0 = str(params.t0)
    with open('params_{}.json'.format(scn_name), 'w') as file:
        json.dump(params, file)
    sim_zip.write('params_{}.json'.format(scn_name))
    os.remove('params_{}.json'.format(scn_name))
    # df_req = inData.requests[['pax_id','origin','destination','treq','dist']]
    # if 'haver_dist' in inData.requests.columns:
        # df_req['haver_dist'] = inData.requests['haver_dist']
    # df_pax = inData.passengers[['VoT','U_car','U_pt','U_bike']]
    # df = pd.concat([df_req, df_pax], axis=1)
    # sim_zip.writestr("requests.csv", df.to_csv()) 
    # sim_zip.writestr("PT_itineraries.csv", inData.pt_itinerary.to_csv())
    # sim_zip.writestr("vehicles.csv", inData.vehicles[['pos','res_wage']].to_csv())
    sim_zip.writestr("platforms.csv", inData.platforms.to_csv())
    sim_zip.writestr("all_pax.csv", all_pax.to_csv())
    evol_micro = init_d2d_dotmap()

    # Day-to-day simulator
    for day in range(params.get('nD', 1)):  # run iterations

        #----- Pre-day -----#
        inData.passengers = mode_preday(inData, params) # mode choice

        #----- Within-day simulator -----#
        if params.get('wd_simulator', 'MaaSSim') == 'MaaSSim':
            sim.make_and_run(run_id=day)  # prepare and SIM
            sim.output()  # calc results
            sim.last_res = sim.res[day].copy() # create a copy of the results - saved later
            del sim.res[day]
        if params.get('wd_simulator', 'MaaSSim') == 'FleetPy':
            # Pre-day work choice
            inData.vehicles = work_preday(inData.vehicles, params)

            # Determine which platform(s) agents can use
            inData.vehicles.platform = inData.vehicles.apply(lambda x: '0;1' if x.ptcp else ';', axis=1)
            inData.passengers.platforms = inData.passengers.apply(lambda x: '0;1' if x.mode_day == 'rs' else ';', axis=1)

            # Generate input csv's for FleetPy
            dtd_result_dir = os.path.join(path, 'temp_res','{}'.format(scn_name))
            if not os.path.exists(dtd_result_dir):
                os.mkdir(dtd_result_dir)
            inData.requests.to_csv(os.path.join(dtd_result_dir,'inData_requests.csv'))
            inData.passengers.to_csv(os.path.join(dtd_result_dir,'inData_passengers.csv')) 
            inData.vehicles.to_csv(os.path.join(dtd_result_dir,'inData_vehicles.csv')) 
            inData.platforms.to_csv(os.path.join(dtd_result_dir,'inData_platforms.csv')) 

            # FleetPy init: conversion from MaaSSim data structure
            fp_run_id = scn_name + '_day_{}'.format(day) # id in FleetPy
            transform_dtd_output_to_wd_input(dtd_result_dir, fleetpy_dir, fleetpy_study_name, network_name, fp_run_id, demand_name, params)

            # Run FleetPy model
            scn_file = os.path.join(fleetpy_dir, "studies", fleetpy_study_name, "scenarios", f"{fp_run_id}.csv")
            run_scenarios(constant_config_file, scn_file)

            # FleetPy results: convert back to MaaSSim structure (simulator object) #TODO: o.a. indicators per platform, expected in-vehicle time, multi-homing vs single-homing
            sim = transform_wd_output_to_d2d_input(sim, fleetpy_dir, fleetpy_study_name, fp_run_id, inData)

        #----- Post-day -----#
        # Determine key KPIs
        drivers_summary = update_d2d_drivers(sim=sim, params=params)
        travs_summary = update_d2d_travellers(sim=sim, params=params)
        
        # Update work experience of job seekers
        exp_df = update_work_exp(inData, drivers_summary)   # number of days work experience
        inData.vehicles.work_exp = exp_df.work_exp

        # Supply-side diffusion of platform information
        res_inf_driver = wom_driver(inData, params=params)
        inData.vehicles.informed = res_inf_driver   # which job seekers are informed about ride-hailing
        inData.vehicles.expected_income = learning_unregist(inData, drivers_summary, params = params) # determine what income do unregistered job seekers expect based on communication with others
        
        # (De-)registration decisions
        res_regist = platform_regist(inData, drivers_summary, params=params)
        inData.vehicles.registered = res_regist.registered
        inData.vehicles.work_exp = res_regist.work_exp
        inData.vehicles.days_since_reg = res_regist.days_since_reg
        # inData.vehicles.expected_income = res_regist.expected_income # TODO: does it need to be included?
        inData.vehicles.pos = fixed_supply.pos
        inData.vehicles.rejected_reg = res_regist.rejected_reg

        # Demand-side diffusion of platform information
        res_inf_trav = wom_trav(inData, travs_summary, params=params)
        inData.passengers.informed = res_inf_trav.informed
        inData.passengers.expected_wait = res_inf_trav.perc_wait

        # Store KPIs of day
        evol_micro = d2d_summary_day(evol_micro, drivers_summary, travs_summary, day)

        # Stop criterion
        if sim.functions.f_stop_crit(sim=sim):
            break

    # Compute aggregated statistics from individual agent results and store both       
    evol_micro, evol_agg = d2d_agg_statistics(evol_micro)
    for data_sup in ['inform', 'regist', 'ptcp', 'perc_inc', 'exp_inc']:
        sim_zip.writestr("d2d_driver_{}.csv".format(data_sup), evol_micro.supply.toDict()[data_sup].to_csv())
    for data_dem in ['inform', 'requests', 'wait_time', 'corr_wait_time', 'perc_wait', 'bike', 'car', 'pt']:
        sim_zip.writestr("d2d_traveller_{}.csv".format(data_dem), evol_micro.demand.toDict()[data_dem].to_csv())
    sim_zip.writestr("d2d_agg_supply.csv", evol_agg.supply.to_csv())
    sim_zip.writestr("d2d_agg_demand.csv", evol_agg.demand.to_csv())

    return sim


if __name__ == "__main__":
    # simulate(make_main_path='..')  # single run
    simulate()  # single run

    from MaaSSim.MaaSSim.utils import test_space

    simulate_parallel(search_space = test_space())
