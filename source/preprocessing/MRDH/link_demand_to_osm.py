# In this script, we determine which OSM nodes are closest to the origin and destination of each trip request.
import pandas as pd
import networkx as nx
import os
import sys
import numpy as np
import osmnx as ox
import geopandas as gpd

current_script_path = os.path.abspath(__file__)
fleetmaas_repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))
sys.path.append(fleetmaas_repo_path)

def load_fleetpy_network(area_name, network_type, return_skim=True, presaved_zones=True):
    '''load FleetPy network files, which have been preprocessed from OSM'''
    network_type_extension = "" if network_type == "drive" else "_{}".format(network_type)
    network_path = os.path.join(fleetmaas_repo_path, "FleetPy", "data", "networks", area_name + network_type_extension)

    def load_graph():
        # Load CSV data
        nodes_df = pd.read_csv(os.path.join(network_path, "base", "nodes.csv"))  # Columns: node_index, pos_x, pos_y
        edges_df = pd.read_csv(os.path.join(network_path, "base", "edges.csv"))  # Columns: from_node, to_node, distance, travel_time

        # Create directed graph
        G = nx.MultiDiGraph()

        # Add nodes
        for _, row in nodes_df.iterrows():
            G.add_node(row["node_index"], pos=(row["pos_x"], row["pos_y"]))
            G.nodes[row["node_index"]]["x"] = row["pos_x"]  # Add longitude
            G.nodes[row["node_index"]]["y"] = row["pos_y"]  # Add latitude

        # Add edges
        for _, row in edges_df.iterrows():
            G.add_edge(row["from_node"], row["to_node"], 
                    length=row["distance"], 
                    travel_time=row["travel_time"])

        # Load GeoJSON files
        nodes_gdf = gpd.read_file(os.path.join(network_path, "base", "nodes_all_infos.geojson"))  # Contains exact node positions
        edges_gdf = gpd.read_file(os.path.join(network_path, "base", "edges_all_infos.geojson"))  # Contains road geometries

        # Extract CRS from GeoDataFrames
        crs = nodes_gdf.crs

        # Add geometries to edges in NetworkX graph
        for _, row in edges_gdf.iterrows():
            u, v = row["from_node"], row["to_node"]
            if G.has_edge(u, v):
                for key in G[u][v]:
                    G[u][v][key]["geometry"] = row["geometry"]  # Store road shape

        # Assign CRS to the graph
        G.graph['crs'] = crs

        # Load zones
        if not presaved_zones:
            zone_path = os.path.join(fleetmaas_repo_path, "source", "preprocessing", "MRDH", "network_zones.geojson")
            load_and_save_zones(zone_path, nodes_gdf)

        return G

    def load_skims():
        '''load the precomputed skim matrices (travel time and distance)'''
        tt_skim = np.load(os.path.join(network_path, "ff", "tables", "nn_fastest_tt.npy"), mmap_mode="r")
        distance_skim = np.load(os.path.join(network_path, "ff", "tables", "nn_fastest_distance.npy"), mmap_mode="r")
        skims = {'tt': tt_skim, 'distance': distance_skim}

        return skims
    
    def load_and_save_zones(zone_system_path, nodes):
        '''load zones and assign zone to each node, then save to csv'''

        # Load the postcode zones as GeoDataFrame
        zones = gpd.read_file(zone_system_path)  # Make sure it has a polygon geometry and a postcode/zone column

        # Ensure both are in the same coordinate reference system
        nodes = nodes.to_crs(zones.crs)

        # Optional: drop nodes without geometry (edge case)
        nodes = nodes.dropna(subset=["geometry"])

        # Spatial join: assign each node to the zone polygon it falls into
        # `how="left"` keeps all nodes, even if they don't match a zone (zone will be NaN)
        nodes_with_zones = gpd.sjoin(nodes, zones, how="left", predicate="within")

        # Save to CSV: node ID and zone ID
        output = nodes_with_zones[["pc4_code"]].copy()
        output["node_id"] = nodes_with_zones.index
        output = output[["node_id", "pc4_code"]]  # Rearrange columns

        # Save to CSV
        output.to_csv("nodes_with_zones.csv", index=False)
    
    if return_skim:
        return load_graph(), load_skims()
    else:
        return load_graph(), None


def coordinate_conversion(rd_x, rd_y):
    '''Convert trip coordinates for MRDH data (RD New coordinates) to WGS84 coordinates'''

    # Convert RD new coordinates to WGS84 coordinates
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:28992", "EPSG:4326")

    # Convert to WGS 84 (Longitude, Latitude)
    lon, lat = transformer.transform(rd_x, rd_y)

    return (lon, lat)


def determine_preprocessed_network_types(study_area, nw_type_extension_dict):
    '''Determine which preprocessed network types are available in FleetPy network folder'''
    
    # List available networks in FleetPy network folder
    network_path = os.path.join(fleetmaas_repo_path, "FleetPy", "data", "networks")
    network_folders = os.listdir(network_path)
    nw_type_extensions = ["" if nw_type == study_area else "_{}".format(nw_type.split("_")[1]) for nw_type in network_folders if nw_type.startswith(study_area)]
    inverse_dict = {v: k for k, v in nw_type_extension_dict.items()}  # Invert the dictionary
    nw_types = [inverse_dict[item] for item in nw_type_extensions]

    return nw_types
