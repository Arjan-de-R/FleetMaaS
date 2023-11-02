import os
import pandas as pd
import numpy as np

def offer_received(row):
    '''determine which platforms give offer to traveller'''
    offers = row.offers.split("|")
    offer_received = []
    for plf in range(len(offers)):
        if offers[plf].split(":")[1] == "": # no (i.e. empty) offer is provided
            offer_received = offer_received + [False]
        else:
            offer_received = offer_received + [True]
    return np.array(offer_received)


def transform_wd_output_to_d2d_input(sim, fleetpy_dir, fleetpy_study_name, fp_run_id, inData):
    '''This function transforms the output files of the within-day model (FleetPy) for the day-to-day (MaaSSim) model'''
    result_dir = os.path.join(fleetpy_dir, 'studies', fleetpy_study_name, 'results', fp_run_id) # where are the results stored

    # 1) Load traveller KPIs
    req_kpis = pd.read_csv(os.path.join(result_dir,'1_user-stats.csv'))
    if not req_kpis.empty:
        req_kpis = pd.read_csv(os.path.join(result_dir,'1_user-stats.csv'), index_col='request_id')
        pax_exp = pd.DataFrame(index = req_kpis.index)
        pax_exp.index.name = 'pax'
        pax_exp['TRAVEL'] = req_kpis.dropoff_time - req_kpis.pickup_time - 30 # total travel time
        pax_exp['WAIT'] = req_kpis.pickup_time - req_kpis.rq_time
        pax_exp['OPERATIONS'] = 30
        pax_exp['detour'] = (req_kpis.dropoff_time - req_kpis.pickup_time - req_kpis.direct_route_travel_time - pax_exp.OPERATIONS) / req_kpis.direct_route_travel_time # relative detour time
        pax_exp['fare'] = req_kpis.fare / 100
        pax_exp['platform'] = req_kpis.operator_id
        pax_exp['LOSES_PATIENCE'] = req_kpis.apply(lambda row: ~offer_received(row), axis=1)
        for col in req_kpis.columns:
            if col.startswith('time occ'):
                pax_exp[col.replace(" ","_")] = req_kpis[col]
    else: 
        pax_exp = pd.DataFrame()
    # Now we add travellers that did not make a request
    all_travs = inData.passengers.index.values
    req_travs = pax_exp.index.values
    noreq_travs = list(set(all_travs) - set(req_travs))
    time_occ_indicators = [s for s in pax_exp.columns if s.startswith("time_occ")]
    no_req_df = pd.DataFrame(np.nan, index=noreq_travs, columns=(['TRAVEL','WAIT','OPERATIONS','detour','fare','platform','LOSES_PATIENCE'] + time_occ_indicators))
    no_req_df['LOSES_PATIENCE'] = no_req_df.apply(lambda row: np.full(len(inData.platforms.index.values), None), axis=1)
    pax_exp = pd.concat([pax_exp,no_req_df]).sort_index()
    sim.last_res.pax_exp = pax_exp.copy() # store in (MaaS)Sim simulator object

    # 2) Load driver KPIs
    # find all columns included in platform output files (i.e. including highest occupancy)
    all_col_names = []
    for plf in np.arange(len(inData.platforms.index)+1): # additional platform is for repositioning (not associated with an actual platform)
        plf_kpis = pd.read_csv(os.path.join(result_dir,'standard_mod-{}_veh_eval.csv'.format(plf)), index_col = 0)
        col_names = plf_kpis.columns.values.tolist()
        all_col_names = all_col_names + list(set(col_names).difference(all_col_names))
       
    aggr_kpis = pd.DataFrame(columns=all_col_names)
    if 'driver_id' in aggr_kpis.columns: # at least a single driver
        aggr_kpis = aggr_kpis.set_index('driver_id')
    for plf in np.arange(len(inData.platforms.index)+1): # additional platform is for repositioning (not associated with an actual platform)
        plf_kpis = pd.read_csv(os.path.join(result_dir,'standard_mod-{}_veh_eval.csv'.format(plf)), index_col = 0)
        if not plf_kpis.empty: # at least one driver for this platform
            # check which columns are missing
            missing_cols = list(set(all_col_names).difference(plf_kpis.columns.values.tolist()))
            plf_kpis[missing_cols] = 0
        else:
            plf_kpis = pd.DataFrame(columns=all_col_names)
        if 'driver_id' in plf_kpis.columns:
            plf_kpis = plf_kpis.set_index('driver_id')
        if plf == len(inData.platforms.index): # the repositioning dataframe, not an actual platform
            if 'km occ 0' in plf_kpis.columns: # at least one of the platforms has at least one driver
                plf_kpis['pickup_dist'] = 0
                plf_kpis['repos_dist'] = plf_kpis['km occ 0']
        else: # corresponding to platform
            if 'km occ 0' in plf_kpis.columns:
                plf_kpis['repos_dist'] = 0
                plf_kpis['pickup_dist'] = plf_kpis['km occ 0']
            
        aggr_kpis = pd.concat([aggr_kpis, plf_kpis])
    
    if not aggr_kpis.empty: # there is at least a single driver
        aggr_kpis = aggr_kpis.groupby('driver_id').sum()
        veh_exp = pd.DataFrame(index = aggr_kpis.index)
        veh_exp['NET_INCOME'] = (aggr_kpis.revenue - aggr_kpis['total variable costs']) / 100
        for col in aggr_kpis.columns:
            if col in ['pickup_dist', 'repos_dist'] or col.startswith("km occ"):
                veh_exp[col] = aggr_kpis[col]
    else:
        veh_exp = aggr_kpis.copy()

    # Now we add drivers that did not work today
    all_drivers = inData.vehicles.index.values
    ptcp_drivers = veh_exp.index.values
    noptcp_drivers = list(set(all_drivers) - set(ptcp_drivers))
    kpis_no_ptcp, dtype_no_ptcp = dict(), dict()
    for col in veh_exp:
        kpis_no_ptcp[col] = np.nan
        dtype_no_ptcp[col] = "float64"
    # kpis_no_ptcp = {'NET_INCOME': np.nan, 'pickup_dist': np.nan, 'repos_dist': np.nan}
    no_ptcp_df = pd.DataFrame.from_dict(kpis_no_ptcp, orient='index').transpose()
    no_ptcp_df = pd.DataFrame(np.repeat(no_ptcp_df.to_numpy(), len(noptcp_drivers), axis=0), columns=no_ptcp_df.columns) # repeat same row for all drivers that did not work
    # no_ptcp_df = no_ptcp_df.astype(dtype= {"NET_INCOME":"float64", "pickup_dist":"float64", "repos_dist":"float64"})
    no_ptcp_df =  no_ptcp_df.astype(dtype=dtype_no_ptcp)
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