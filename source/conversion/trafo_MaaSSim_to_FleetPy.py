import os
import datetime as dt
import pandas as pd

from FleetPy.src.misc.globals import *


def _create_seconds_of_day(dt_str):
    hour, minute, second =  [int(x) for x in dt_str.split(" ")[1].split(":")]
    return 3600 * hour + 60 * minute + second


def transform_dtd_output_to_wd_input(dtd_result_dir, fleetpy_dir, fleetpy_study_name, nw_name, new_wd_scenario_name, demand_name, d2d_params):
    """This function transforms the output files of the day-to-day model for the within-day (FleetPy) model.

    :param dtd_result_dir: result directory of last day-to-day iteration
    :param fleetpy_data_dir: FleetPy main directory
    :param fleetpy_study_name: name of study
    :param nw_name: network name
    :param new_wd_scenario_name: scenario name of next within-day simulation
    :param demand_name: demand name
    :param d2d_params: day-to-day params, incl. start and end time of all days
    """
    # 0) convert MaaSSim node id to FleetPy node id
    fp_nodes = pd.read_csv(r'FleetPy\data\networks\Delft\base\nodes.csv')  #TODO: make relative path
    trafo_dict = fp_nodes[['node_index', 'source_node_id']].set_index('source_node_id').to_dict()['node_index']

    # 1) create demand data set
    rq_f = os.path.join(dtd_result_dir, "inData_requests.csv")
    rq_df = pd.read_csv(rq_f, index_col=0)
    rq_df.index.name = "rq_id"
    pax_f = os.path.join(dtd_result_dir, "inData_passengers.csv")
    pax_df = pd.read_csv(pax_f, index_col=0)
    c_df = pd.merge(rq_df, pax_df, left_on="pax_id", right_index=True)
    f_df = c_df.loc[c_df["platforms"] != ';'].reset_index()
    f_df['start'] = f_df['origin'].apply(lambda x: trafo_dict[x])
    f_df['end'] = f_df['destination'].apply(lambda x: trafo_dict[x])
    fpy_rq_df = f_df[["rq_id", "treq", "start", "end", "platforms", "VoT"]]
    fpy_rq_df["rq_time"] = fpy_rq_df.apply(lambda x: _create_seconds_of_day(x["treq"]), axis=1)
    fpy_rq_df.rename({"rq_id": "request_id", "VoT": "value_of_time"}, axis=1, inplace=True) # rename
    fpy_rq_df.sort_values("rq_time", inplace=True)

    nodes_df = pd.read_csv(os.path.join(fleetpy_dir, "data", "networks", nw_name, "base", "nodes.csv"))
    if not "source_node_id" in nodes_df.columns:
        raise EnvironmentError("Network file has no information of source ids (source_node_id entry missing in nodes.csv) -> no conversion possible!")
    nodes_df.set_index("source_node_id", inplace=True)
    source_to_node_id = nodes_df["node_index"].to_dict()
    fpy_rq_f = os.path.join(fleetpy_dir, "data", "demand", demand_name)
    if not os.path.isdir(fpy_rq_f):
        os.mkdir(fpy_rq_f)
    fpy_rq_f = os.path.join(fpy_rq_f, "matched")
    if not os.path.isdir(fpy_rq_f):
        os.mkdir(fpy_rq_f)
    fpy_rq_f = os.path.join(fpy_rq_f, nw_name)
    if not os.path.isdir(fpy_rq_f):
        os.mkdir(fpy_rq_f)
    # fpy_rq_f = os.path.join(fleetpy_dir, "data", "demand", fleetpy_study_name, nw_name, f"{new_wd_scenario_name}.csv")
    fpy_rq_f = os.path.join(fleetpy_dir, "data", "demand", demand_name, "matched", nw_name, f"{new_wd_scenario_name}.csv")
    try:
        fpy_rq_df.to_csv(fpy_rq_f, index=False)
    except:
        raise NotImplementedError

    # 2) create driver/vehicle data file
    # TODO # specification of input file and codes for initialization (and rest)
    driver_f = os.path.join(dtd_result_dir, "inData_vehicles.csv")
    driver_df = pd.read_csv(driver_f, index_col=0)
    driver_df.index.name = "driver_id"
    print(driver_df.head())
    #driver_id,veh_type,possible_operators,start_node,operating_times
    fp_driver_df_list = []
    for driver_id, driver_row in driver_df.iterrows():
        fp_driver_df_list.append({
            "driver_id" : driver_id,
            "start_node" : source_to_node_id[driver_row["pos"]],  
            "possible_operators" : driver_row["platform"],   # TODO how is encoding for multiple operators?
            "operating_times" : f"{driver_row['shift_start']};{driver_row['shift_end']}",
            "veh_type" : "default_vehtype" # TODO?
        })
    fp_driver_df = pd.DataFrame(fp_driver_df_list)
    fp_driver_f = os.path.join(fleetpy_dir, "data", "fleetctrl", "freelancer_drivers")
    if not os.path.isdir(fp_driver_f):
        os.mkdir(fp_driver_f)  
    fp_driver_f = os.path.join(fp_driver_f, nw_name)  
    if not os.path.isdir(fp_driver_f):
        os.mkdir(fp_driver_f) 
    fp_driver_f_name = f"driver_{new_wd_scenario_name}.csv"
    fp_driver_f = os.path.join(fp_driver_f, fp_driver_f_name)  
    fp_driver_df.to_csv(fp_driver_f, index=False)
    print(fp_driver_df.head())
    #driver_id,veh_type,possible_operators

    # 3) create scenario input file
    start_time = _create_seconds_of_day(d2d_params.t0)
    end_time = start_time + d2d_params.simTime * 3600
    platform_df = pd.read_csv(os.path.join(dtd_result_dir, "inData_platforms.csv"))
    nr_platforms = platform_df.shape[0]
    op_min_fares = ";".join([str(x) for x in platform_df["min_fare"].values])
    op_base_fares = ";".join([str(x) for x in platform_df["base_fare"].values])
    op_dis_fares = ";".join([str(x/1000.0) for x in platform_df["fare"].values])
    op_comm_rates = ";".join([str(x) for x in platform_df["comm_rate"].values])
    op_max_detour_time_factor = ";".join([str(x) for x in platform_df["max_rel_detour"].values])
    op_max_wait_time = ";".join([str(x) for x in platform_df["max_wait_time"].values])
    op_vr_control_func_dict = "|".join([str(x) for x in platform_df["match_obj"].values])

    print(platform_df.head())
    
    sc_df_list = [{
        G_SCENARIO_NAME : new_wd_scenario_name,
        G_SIM_START_TIME: start_time,
        G_SIM_END_TIME: end_time,
        G_RQ_FILE : f"{new_wd_scenario_name}.csv",
        G_NR_OPERATORS : nr_platforms,
        G_PLAT_DRIVER_FILE : fp_driver_f_name,
        G_OP_FARE_B : op_base_fares,
        G_OP_FARE_D : op_dis_fares,
        G_OP_FARE_MIN : op_min_fares,
        G_OP_PLAT_COMMISION : op_comm_rates,
        G_OP_MAX_DTF: op_max_detour_time_factor,
        G_OP_MAX_WT: op_max_wait_time,
        G_OP_VR_CTRL_F: op_vr_control_func_dict
    }]
    
    sc_df = pd.DataFrame(sc_df_list)
    sc_df.to_csv(os.path.join(fleetpy_dir, "studies", fleetpy_study_name, "scenarios", f"{new_wd_scenario_name}.csv"), index=False)
    print(sc_df)


# testing
if __name__ == '__main__':
    FleetPy_Path = os.path.join( os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "FleetPy")
    fleetpy_study_name = "maasim_fleetpy_trb24"
    dtd_result_dir = os.path.join(FleetPy_Path, "studies", fleetpy_study_name, "Input_MaaSSim")
    fleetpy_dir = FleetPy_Path
    nw_name = "delft"
    new_wd_scenario_name = "test_iteration_1"
    transform_dtd_output_to_wd_input(dtd_result_dir, fleetpy_dir, fleetpy_study_name, nw_name, new_wd_scenario_name)