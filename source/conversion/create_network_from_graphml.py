import sys
import os
import networkx as nx
from shapely.geometry import Point, LineString
import geopandas as gpd

dev_p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(dev_p)

from FleetPy.src.preprocessing.networks.network_manipulation import FullNetwork
from FleetPy.src.preprocessing.networks.create_travel_time_tables import create_travel_time_table


def create_network_from_graphml(graphml_file, network_name):
    """ this function reads the graphml file and creates fleetpy network files
    output files are stored at fleetpy_dir/data/networks/{network_name}
    
    :param network_name: folder name of network (output folder files will be stored here)
    :param graphml_file: path to the graphml_file"""
    
    graph = nx.read_graphml(graphml_file)
    # read nodes
    nodes_df_list = []
    node_osmid_to_id = {}
    c_id = 0
    for node in graph.nodes:
        nodes_df_list.append({
            "pos_x" : graph.nodes[node]["x"],
            "pos_y" : graph.nodes[node]["y"],
            "source_node_id" : graph.nodes[node]["osmid"],
            "is_stop_only" : False,
            "geometry" : Point(float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"]))
        })
        node_osmid_to_id[graph.nodes[node]["osmid"]] = c_id
        c_id += 1
    nodes_gdf = gpd.GeoDataFrame(nodes_df_list)
    # read edges
    edges_df_list = []
    edge_osmid_to_id = {}
    c_id = 0
    for edge in graph.edges:
        o_node_osmid = edge[0]
        d_node_osmid = edge[1]
        o_node_id = node_osmid_to_id[o_node_osmid]
        d_node_id = node_osmid_to_id[d_node_osmid]
        o_node = nodes_gdf.iloc[o_node_id]
        d_node = nodes_gdf.iloc[d_node_id]
        #print(o_node, d_node)
        
        source_edge_id = graph.edges[edge]["osmid"]
        length = graph.edges[edge]["length"]
        # infer edge speeds if not data given
        if len(graph.edges[edge]["highway"].split(",")) > 1: # if link has different kind of lanes
            graph.edges[edge]["highway"] = graph.edges[edge]["highway"].replace("[","").replace("]","").replace("'","").split(",")[0] # assume first lane type for all
        try:
            speed = graph.edges[edge]["maxspeed"]
            if len(speed.split(",")) > 1:
                speed = speed.split(",")[0][2:-1:]
                #print("was list!", speed)
        except KeyError:
            if graph.edges[edge]["highway"] in ["residential", "road"]:
                speed = 30
            elif graph.edges[edge]["highway"] == "living_street":
                speed = 15
            elif graph.edges[edge]["highway"] in ["unclassified", "primary", "secondary", "tertiary", "primary_link", "secondary_link", "tertiary_link"]:
                speed = 50
            elif graph.edges[edge]["highway"] in ["motorway_link", "motorway"]:
                speed = 100
            elif graph.edges[edge]["highway"] in ["trunk_link", "trunk"]:
                speed = 80
            else:
                raise KeyError
        travel_time = float(length)/float(speed)*3.6
        
        if graph.edges[edge].get("geometry"):
            geo = graph.edges[edge].get("geometry")
            geo = geo.split("(")[1]
            geo = geo.split(")")[0]
            coord_list = []
            for x_y_str in geo.split(","):
                xy = x_y_str.split(" ")
                if len(xy) == 2:
                    x, y = xy[0], xy[1]
                if len(xy) == 3:
                    x, y = xy[1], xy[2]
                coord_list.append( (float(x), float(y)) )
            geo = LineString(coord_list)
        else:
            geo = LineString([o_node["geometry"], d_node["geometry"]])
        
        edges_df_list.append({
            "from_node" : o_node_id,
            "to_node" : d_node_id,
            "distance" : length,
            "travel_time" : travel_time,
            "source_edge_id" : source_edge_id,
            "geometry" : geo
        })
        

    edges_gdf = gpd.GeoDataFrame(edges_df_list)
    
    nw = FullNetwork(None, nodes_gdf=nodes_gdf, edges_gdf=edges_gdf)
    
    nw.storeNewFullNetwork(os.path.join(dev_p, "FleetMaaS", "FleetPy", "data", "networks"), network_name)
    
    nw.plotNetwork()

    network_dir = os.path.join(dev_p, "FleetMaaS", "FleetPy", "data", "networks", network_name)
    create_travel_time_table(network_dir, scenario_time=None, save_npy=True, save_csv=False)
    

def get_init_zone_id(row, gpd_zones):
    '''function for determining zone id of specific node'''
    for zone_id, zone_attrib in gpd_zones.iterrows():
        if row.geometry.within(zone_attrib.geometry):
            return zone_id


if __name__ == "__main__":
    graphml_file = r"C:\Users\ge37ser\Documents\Coding\TUM_VT_FleetSimulation\tum-vt-fleet-simulation\FleetPy\studies\maasim_fleetpy_trb24\preprocessing\delft_test_network\Delft.graphml"
    network_name = "delft"
    
    create_network_from_graphml(graphml_file, network_name)