################################################################################
# Module: runners.py
# Description: Wrappers to prepare and run simulations
# Rafal Kucharski @ TU Delft
################################################################################


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
from src.conversion.create_network_from_graphml import *
from src.conversion.trafo_FleetPy_to_MaaSSim import *
from src.conversion.trafo_MaaSSim_to_FleetPy import *
import os.path
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

    stamp['dt'] = str(pd.Timestamp.now()).replace('-','').replace('.','').replace(' ','')

    filename = ''
    for key, value in stamp.items():
        filename += '-{}_{}'.format(key, value)
    filename = re.sub('[^-a-zA-Z0-9_.() ]+', '', filename)

    sim = simulate(inData=_inData, params=_params, logger_level=logging.WARNING, filename = filename)

    print(filename, pd.Timestamp.now(), 'end')
    return 0


def simulate_parallel(config="../data/config/parallel.json", inData=None, params=None, search_space=None, **kwargs):
    if inData is None:  # otherwise we use what is passed
        from MaaSSim.MaaSSim.data_structures import structures
        inData = structures.copy()  # fresh data
    if params is None:
        params = get_config(config, root_path = kwargs.get('root_path'))  # load from .json file

    if len(inData.G) == 0:  # only if no graph in input
        inData = load_G(inData, params, stats=True)  # download graph for the 'params.city' and calc the skim matrices
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

    # Generate requests - either based on a distribution or taken from Albatross - and corresponding passenger data
    if params.get('albatross', False):
        inData = load_albatross_proc(inData, params, avg_speed = True)
        inData.requests = inData.requests.drop(['orig_geo', 'dest_geo', 'origin_y', 'origin_x', 'destination_y', 'destination_x', 'time'], axis = 1)
        inData = sample_from_alba(inData, params)
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

    # Generate information available to travellers at the start of the simulation
    inData.passengers['informed'] = np.random.rand(len(inData.passengers)) < params.evol.travellers.inform.prob_start
    inData.passengers['expected_wait'] = params.evol.travellers.inform.start_wait
    
    # Generate pool of job seekers
    fixed_supply = generate_vehicles_d2d(inData, params)
    inData.vehicles = fixed_supply.copy()
    
    # Determine which platform agents can use
    inData.vehicles.platform = inData.vehicles.apply(lambda x: 0, axis = 1)  # TODO: allow for multiple platforms
    inData.passengers.platforms = inData.passengers.apply(lambda x: [0], axis = 1)
    inData.requests['platform'] = inData.requests.apply(lambda row: inData.passengers.loc[row.name].platforms[0], axis = 1)

    # Set properties of platform(s)
    inData.platforms = pd.concat([inData.platforms,pd.DataFrame(columns=['base_fare','comm_rate','min_fare'])])
    inData.platforms = initialize_df(inData.platforms)
    inData.platforms.loc[0]=[params.platforms.fare,'Uber',30,params.platforms.base_fare,params.platforms.comm_rate,params.platforms.min_fare,]

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
        graphml_file = params.paths.G
        network_name = params.city.split(",")[0]
        create_network_from_graphml(graphml_file, network_name) # create FleetPy network files

    # Where will results be stored
    filename = kwargs.get('filename')
    if path is None:
        path = os.getcwd()
    sim_zip = zipfile.ZipFile(os.path.join(path, '{}.zip'.format(filename)), 'w')
    params.t0 = str(params.t0)
    with open('params_{}.json'.format(filename), 'w') as file:
        json.dump(params, file)
    sim_zip.write('params_{}.json'.format(filename))
    os.remove('params_{}.json'.format(filename))
    df_req = inData.requests[['pax_id','origin','destination','treq','dist','haver_dist']]
    df_pax = inData.passengers[['VoT','U_car','U_pt','U_bike']]
    df = pd.concat([df_req, df_pax], axis=1)
    sim_zip.writestr("requests.csv", df.to_csv())
    # sim_zip.writestr("PT_itineraries.csv", inData.pt_itinerary.to_csv())
    sim_zip.writestr("vehicles.csv", inData.vehicles[['pos','res_wage']].to_csv())
    sim_zip.writestr("platforms.csv", inData.platforms.to_csv())
    sim_zip.writestr("all_pax.csv", all_pax.to_csv())
    evol_micro = init_d2d_dotmap()

    # Day-to-day simulator
    for day in range(params.get('nD', 1)):  # run iterations

        #----- Pre-day -----#
        inData.passengers = mode_preday(inData, params) # mode choice

        #----- Within-day simulation -----#
        if params.get('wd_simulator', 'MaaSSim') == 'MaaSSim':
            sim.make_and_run(run_id=day)  # prepare and SIM
            sim.output()  # calc results
            sim.last_res = sim.res[day].copy() # create a copy of the results - saved later
            del sim.res[day]
        if params.get('wd_simulator', 'MaaSSim') == 'FleetPy':
            # FleetPy init: conversion from MaaSSim data structure
            dtd_result_dir = 9999 #TODO
            fleetpy_dir = os.path.join(path, 'FleetPy')
            fleetpy_study_name = params.get('study_name', 'MaaSSim_FleetPy')
            new_wd_scenario_name = 9999 #TODO
            transform_dtd_output_to_wd_input(dtd_result_dir, fleetpy_dir, fleetpy_study_name, network_name, new_wd_scenario_name)

            # Run FleetPy model
            # constant_config_file: setup operator 1 = hailing, op_2 = pooling (dep. on scenario), max. waiting time (everything that will remain constant), SCENARIO-SPECIFIC
            # new_wd_scenario_name: everything that changes from sim to sim
            run_scenarios(constant_config_file, new_wd_scenario_name, n_parallel_sim=1, n_cpu_per_sim=1, evaluate=1, log_level="info",
                  keep_old=False, continue_next_after_error=False)

            # FleetPy results

            # FleetPy conversion: convert results back to MaaSSim datastructure

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
