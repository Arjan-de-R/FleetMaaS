import os
import sys
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point, LineString
import math
import osmnx as ox

dev_p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(dev_p)
sys.path.append(os.path.join(dev_p, "FleetPy", "src", "preprocessing", "networks"))

from FleetPy.src.preprocessing.networks.network_manipulation import FullNetwork
from FleetPy.src.preprocessing.networks.create_travel_time_tables import create_travel_time_table

def create_network_from_osm(study_area, study_area_osm_name, network_type, mode_speeds):
    """ this function loads the OpenStreetMap (OSM) network and creates fleetpy network files
    output files are stored at fleetpy_dir/data/networks/{network_name}
    
    :param network_name: folder name of network (output folder files will be stored here)
    :param graphml_file: path to the graphml_file"""

    # import the graph and keep largest strongly connected component
    graph = ox.graph_from_place(study_area_osm_name, network_type=network_type)
    graph = ox.truncate.largest_component(graph, strongly=True)
    
    # read nodes
    nodes_df_list = []
    node_osmid_to_id = {}
    c_id = 0
    for node in graph.nodes:
        nodes_df_list.append({
            "pos_x" : graph.nodes[node]["x"],
            "pos_y" : graph.nodes[node]["y"],
            "source_node_id" : node,
            "is_stop_only" : False,
            "geometry" : Point(float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"]))
        })
        node_osmid_to_id[node] = c_id
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
        
        if isinstance(graph.edges[edge]["osmid"], list):
            source_edge_id = graph.edges[edge]["osmid"][0]  # Keep only the first OSM ID
        else:
            source_edge_id = graph.edges[edge]["osmid"]
        length = graph.edges[edge]["length"]
        # infer edge speeds if not data given
        if isinstance(graph.edges[edge]["highway"], list): # if link has different kind of lanes
            graph.edges[edge]["highway"] = graph.edges[edge]["highway"][0] # assume first lane type for all
        if network_type == "drive":
            try:
                speed = graph.edges[edge]["maxspeed"]
                if isinstance(speed, list):
                    speed = speed[0]
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
                elif graph.edges[edge]["highway"] == "busway":
                    speed = 0
                else:
                    raise KeyError
            speed = float(speed) * mode_speeds.get('car_congestion_factor', 1)
        elif network_type == "bike":
            speed = mode_speeds.get('bike', 4.166666) * 3.6
        elif network_type == "walk":
            speed = mode_speeds.get('walk', 1.3888889) * 3.6
        else:
            raise ValueError("Invalid network type")
        try:
            travel_time = float(length)/float(speed)*3.6
        except ZeroDivisionError:
            travel_time = math.inf
        
        if graph.edges[edge].get("geometry"):
            geo = graph.edges[edge].get("geometry")
        #     geo = geo.split("(")[1]
        #     geo = geo.split(")")[0]
        #     coord_list = []
        #     for x_y_str in geo.split(","):
        #         xy = x_y_str.split(" ")
        #         if len(xy) == 2:
        #             x, y = xy[0], xy[1]
        #         if len(xy) == 3:
        #             x, y = xy[1], xy[2]
        #         coord_list.append( (float(x), float(y)) )
        #     geo = LineString(coord_list)
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
    
    network_type_extension = "" if network_type == "drive" else "_{}".format(network_type)
    nw.storeNewFullNetwork(os.path.join(dev_p, "FleetPy", "data", "networks"), study_area + network_type_extension)
    
    nw.plotNetwork()

    network_dir = os.path.join(dev_p, "FleetPy", "data", "networks", study_area + network_type_extension)
    create_travel_time_table(network_dir, scenario_time=None, save_npy=True, save_csv=False)


if __name__ == "__main__":
    study_area = "MRDH"
    study_area_osm_name = "Metropolitan Region Rotterdam The Hague"
    network_type = "drive" #"drive"
    mode_speeds = {'car_congestion_factor': 0.9, "bike": 20 / 3.6, "walk": 5 / 3.6}
    
    create_network_from_osm(study_area, study_area_osm_name, network_type, mode_speeds)