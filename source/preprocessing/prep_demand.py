### Determining fastest travel time based on FleetPy network prior to simulation ###

import pandas as pd
import numpy as np
import datetime
import os
path = os.getcwd()
MAASSIM_path = os.path.join(path, 'MaaSSim')
FLEETPY_path = os.path.join(path, 'FleetPy')

path_demand = os.path.join(MAASSIM_path,"data", "demand", "Amsterdam", "albatross", "preprocessed.csv")
path_network_tt = os.path.join(FLEETPY_path, "data", "networks", "Amsterdam", "ff", "tables", "nn_fastest_tt.npy")
path_network_dist = os.path.join(FLEETPY_path, "data", "networks", "Amsterdam", "ff", "tables", "nn_fastest_distance.npy")
path_nodes = os.path.join(FLEETPY_path, "data", "networks", "Amsterdam", "base", "nodes.csv")

reqs = pd.read_csv(path_demand)
tt = np.load(path_network_tt)
dist = np.load(path_network_dist)
node_list = pd.read_csv(path_nodes)

def determine_shortest_path(row):
    fp_origin = node_list[node_list.source_node_id == row.origin].node_index.values
    fp_dest = node_list[node_list.source_node_id == row.destination].node_index.values
    ttrav = tt[fp_origin,fp_dest][0]
    distance = dist[fp_origin,fp_dest][0]
    return [ttrav, distance]

reqs['ttrav'] = reqs.apply(lambda row: determine_shortest_path(row)[0], axis=1)
reqs['ttrav'] = reqs.apply(lambda row: datetime.timedelta(days=0, seconds=row.ttrav), axis=1)
reqs['dist'] = reqs.apply(lambda row: determine_shortest_path(row)[1], axis=1)
reqs.to_csv('preprocessed.csv',index=False)


