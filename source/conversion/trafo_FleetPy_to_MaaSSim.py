import os
import pandas as pd
import numpy as np

def transform_wd_output_to_d2d_input(sim, fleetpy_dir, fleetpy_study_name, fp_run_id, inData):
    '''This function transforms the output files of the within-day model (FleetPy) for the day-to-day (MaaSSim) model'''
    result_dir = os.path.join(fleetpy_dir, 'studies', fleetpy_study_name, 'results', fp_run_id) # where are the results stored

    # 1) Load traveller KPIs # TODO: what if there are no users?
    req_kpis = pd.read_csv(os.path.join(result_dir,'1_user-stats.csv'), index_col = 'request_id')
    pax_exp = pd.DataFrame(index = req_kpis.index)
    pax_exp.index.name = 'pax'
    pax_exp['TRAVEL'] = req_kpis.dropoff_time - req_kpis.pickup_time # total travel time
    pax_exp['WAIT'] = req_kpis.pickup_time - req_kpis.rq_time
    pax_exp['OPERATIONS'] = 60 # 30 sec both for accessing and egressing the vehicle
    pax_exp['detour'] = req_kpis.dropoff_time - req_kpis.pickup_time - req_kpis.direct_route_travel_time - pax_exp.OPERATIONS # detour time
    pax_exp['fare'] = req_kpis.fare / 100
    pax_exp['platform'] = req_kpis.operator_id
    pax_exp['LOSES_PATIENCE'] = req_kpis.apply(lambda x: 999 if x.operator_id == '' else 0, axis=1) # used later in determining which travellers are rejected by platform
    pax_exp['NO_REQUEST'] = False # True if another mode is chosen prior to the day
    pax_exp['OTHER_MODE'] = False # True if request is made but received offers are rejected
    # Now we add travellers that did not make a request
    all_travs = inData.passengers.index.values
    req_travs = pax_exp.index.values
    noreq_travs = list(set(all_travs) - set(req_travs))
    no_req_df = pd.DataFrame(np.nan, index=noreq_travs, columns=['TRAVEL','WAIT','OPERATIONS','detour','fare','platform','LOSES_PATIENCE'])
    no_req_df['NO_REQUEST'], no_req_df['OTHER_MODE'] = [True, False]
    pax_exp = pd.concat([pax_exp,no_req_df]).sort_index()
    sim.last_res.pax_exp = pax_exp.copy() # store in MaaSSim simulator object

    # 2) Load driver KPIs
    # find all columns included in platform output files (i.e. including highest occupancy)
    all_col_names = []
    for plf in np.arange(len(inData.platforms.index)+1): # additional platform is for multi-homer repositioning (not associated with an actual platform)
        plf_kpis = pd.read_csv(os.path.join(result_dir,'standard_mod-{}_veh_eval.csv'.format(plf)), index_col = 0)
        col_names = plf_kpis.columns.values.tolist()
        all_col_names = all_col_names + list(set(col_names).difference(all_col_names))
       
    aggr_kpis = pd.DataFrame(columns=all_col_names)
    aggr_kpis = aggr_kpis.set_index('driver_id')
    for plf in np.arange(len(inData.platforms.index)+1): # additional platform is for multi-homer repositioning (not associated with an actual platform)
        plf_kpis = pd.read_csv(os.path.join(result_dir,'standard_mod-{}_veh_eval.csv'.format(plf)), index_col = 0)
        if not plf_kpis.empty: # at least one driver for this platform
            # check which columns are missing
            missing_cols = list(set(all_col_names).difference(plf_kpis.columns.values.tolist()))
            plf_kpis[missing_cols] = 0
            plf_kpis = plf_kpis.set_index('driver_id')
        else:
            plf_kpis = pd.DataFrame(columns=all_col_names)
        aggr_kpis = pd.concat([aggr_kpis, plf_kpis])
    aggr_kpis = aggr_kpis.groupby('driver_id').sum()
    veh_exp = pd.DataFrame(index = aggr_kpis.index)
    veh_exp['NET_INCOME'] = (aggr_kpis.revenue - aggr_kpis['total variable costs']) / 100

    # Now we add drivers that did not work today
    all_drivers = inData.vehicles.index.values
    ptcp_drivers = veh_exp.index.values
    noptcp_drivers = list(set(all_drivers) - set(ptcp_drivers))
    kpis_no_ptcp = {'NET_INCOME': np.nan}
    no_ptcp_df = pd.DataFrame.from_dict(kpis_no_ptcp, orient='index').transpose()
    no_ptcp_df = pd.DataFrame(np.repeat(no_ptcp_df.to_numpy(), len(noptcp_drivers), axis=0), columns=no_ptcp_df.columns) # repeat same row for all drivers that did not work
    no_ptcp_df = no_ptcp_df.astype(dtype= {"NET_INCOME":"float64"})
    no_ptcp_df['driver_id'] = noptcp_drivers
    no_ptcp_df = no_ptcp_df.set_index('driver_id')
    veh_exp = pd.concat([veh_exp,no_ptcp_df]).sort_index()
    veh_exp.index.name = 'veh'
    veh_exp['OUT'] = inData.vehicles.apply(lambda row: ~row.ptcp, axis=1)
    veh_exp['FORCED_OUT'] = False # TODO: implement registration cap to work with multiple platforms
    sim.last_res.veh_exp = veh_exp.copy()

    # 3) Save inData to simulator object
    sim.vehicles = inData.vehicles.copy()
    sim.passengers = inData.passengers.copy()
    sim.requests = inData.requests.copy()
    sim.platforms = inData.platforms.copy()
    sim.inData = inData.copy()

    return sim