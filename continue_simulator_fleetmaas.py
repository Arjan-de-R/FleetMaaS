################################################################################
# Module: continue_simulator_fleetmaas.py
# Description: Wrappers to prepare and run simulations - continuing previously terminated simulation
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


def continue_simulate_parallel(config="MaaSSim/data/config/parallel.json", inData=None, params=None, search_space=None, **kwargs):
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

    if len(inData.G) == 0:  # only if no graph in input
        inData = load_G(inData, params, stats=True)  # download graph for the 'params.city' and calc the skim matrices

    # Set random seeds used throughout the simulation
    np.random.seed(params.repl_id)
    random.seed(params.repl_id)
    
    # Load path
    if path is None:
        path = os.getcwd()

    # Prepare schedule for shared rides and the within-day simulator
    # inData = prep_shared_rides(inData, params.shareability)  # prepare schedules
    if params.paths.get('fleetpy_config', False): # initialise FleetPy
        fleetpy_dir = os.path.join(path, 'FleetPy')
        fleetpy_study_name = params.get('study_name', 'MaaSSim_FleetPy')
        config_file = params.paths.fleetpy_config
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
        inData.nodes['geometry'] = geopandas.points_from_xy(inData.nodes['x'],inData.nodes['y'])
        zones = geopandas.read_file(os.path.join(fleetpy_dir, "data", "zones", zone_name, "polygon_definition.geojson"))
        inData.nodes["zone_id"] = inData.nodes.apply(lambda row: get_init_zone_id(row, zones), axis=1)
        constant_config_file = os.path.join(fleetpy_dir,'studies','{}'.format(fleetpy_study_name),'scenarios','{}'.format(config_file))
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
    result_path = os.path.join(path, 'results', scn_name)
    
    params.t0 = params.t0.to_pydatetime().strftime('%Y-%m-%d %H:%M:%S')

    # Load inData.vehicles, inData.passengers, inData.platforms
    pax_properties = pd.read_csv(os.path.join(result_path, '1_pax-properties.csv'), index_col=0)
    driver_properties = pd.read_csv(os.path.join(result_path, '2_driver-properties.csv'),index_col=0)
    inData.platforms = pd.read_csv(os.path.join(result_path, '3_platform-properties.csv'))
    
    # Initialise convergence
    d2d_conv = pd.DataFrame()

    ### Day-to-day simulator

    # First decide which was the last day based on the d2d_conv
    conv_indic = pd.read_csv(os.path.join(result_path,'5_conv-indicators.csv'))
    last_day = conv_indic.shape[0] - 2
    d2d_conv = pd.concat([d2d_conv, conv_indic])
    
    # First we need to test whether sim has not converged already, if not, we continue the simulation from the next day
    determine_convergence(inData, d2d_conv, params, scn_name, last_day)

    # First open the output files from the last day, then create the inData df's
    df_paxx = pd.read_csv(os.path.join(result_path, 'day_{}_travs.csv'.format(last_day)), index_col=0)
    df_drivers = pd.read_csv(os.path.join(result_path, 'day_{}_drivers.csv'.format(last_day)), index_col=0)
    inData.passengers = pd.concat([pax_properties, df_paxx], axis=1)
    inData.requests = pax_properties.reset_index()[['origin','destination','treq','ttrav','dist','pax_id']]
    inData.requests.ttrav = pd.to_timedelta(inData.requests.ttrav)
    inData.passengers.drop(['origin','destination','treq','ttrav','dist','pax_id'],axis=1, inplace=True)
    inData.vehicles = pd.concat([driver_properties,df_drivers], axis=1)

    # Convert dataframe columns
    if inData.platforms.shape[0] > 1:
        inData.passengers['registered'] = inData.passengers.apply(lambda row: np.array([row.registered_0, row.registered_1]), axis=1)
        inData.passengers['expected_wait'] = inData.passengers.apply(lambda row: np.array([row.expected_wait_0, row.expected_wait_1]), axis=1)
        inData.passengers['expected_ivt'] = inData.passengers.apply(lambda row: np.array([row.expected_ivt_0, row.expected_ivt_1]), axis=1)
        inData.passengers['expected_km_fare'] = inData.passengers.apply(lambda row: np.array([row.expected_km_fare_0, row.expected_km_fare_1]), axis=1)
        inData.vehicles['registered'] = inData.vehicles.apply(lambda row: np.array([row.registered_0, row.registered_1]), axis=1)
        inData.vehicles['expected_income'] = inData.vehicles.apply(lambda row: np.array([row.expected_income_0, row.expected_income_1]), axis=1)
    else:
        inData.passengers['registered'] = inData.passengers.apply(lambda row: np.array([row.registered_0]), axis=1)
        inData.passengers['expected_wait'] = inData.passengers.apply(lambda row: np.array([row.expected_wait_0]), axis=1)
        inData.passengers['expected_ivt'] = inData.passengers.apply(lambda row: np.array([row.expected_ivt_0]), axis=1)
        inData.passengers['expected_km_fare'] = inData.passengers.apply(lambda row: np.array([row.expected_km_fare_0]), axis=1)
        inData.vehicles['registered'] = inData.vehicles.apply(lambda row: np.array([row.registered_0]), axis=1)
        inData.vehicles['expected_income'] = inData.vehicles.apply(lambda row: np.array([row.expected_income_0]), axis=1)
    inData.passengers['platforms'] = 0
    inData.vehicles['platform'] = 0
    inData.vehicles['shift_start'] = 0
    inData.vehicles['shift_end'] = 86400
    inData.vehicles['rejected_reg'] = False
    inData.passengers['pos'] = inData.requests.origin
    inData.passengers['zone_id'] = inData.passengers.apply(lambda x: inData.nodes.zone_id.loc[x.pos], axis=1)
    fixed_supply = inData.vehicles[['pos','shift_start']].copy()

    # Load random seeds
    set_random_states(result_path)

    for day in range(last_day+1, params.get('nD', 1)):  # run iterations

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
        dem_df, sup_df = d2d_summary_day(inData, drivers_summary, travs_summary)
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
        
        if determine_convergence(inData, d2d_conv, params, scn_name, day):
            break
        
        # Save random states
        save_random_states(result_path)

        del drivers_summary, travs_summary, dem_df, sup_df

    return sim


if __name__ == "__main__":
    # simulate(make_main_path='..')  # single run
    simulate()  # single run

    from MaaSSim.src_MaaSSim.utils import test_space

    simulate_parallel(search_space = test_space())
