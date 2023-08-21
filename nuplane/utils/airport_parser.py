from __future__ import annotations


from pathlib import Path

from xplane_airports.gateway import scenery_pack
from xplane_apt_convert import ParsedAirport

import networkx as nx
import geopandas as gpd

from nuplane.utils.transform import haversine_distance


def parse_airport_taxi_network(airport_id):
    export_path = f'data/{airport_id}/'

    # Check if the file exists:
    if not Path(export_path + f'{airport_id}.linear_features.shp').is_file():
        # Download scenary pack
        recommended_pack = scenery_pack(airport_id)
        apt = recommended_pack.apt

        # Parse the data
        p_apt = ParsedAirport(apt, bezier_resolution=3)

        # Export
        Path(export_path).mkdir(parents=True, exist_ok=True)

        p_apt.export(
            export_path + f'{airport_id}.shp',
            driver='ESRI Shapefile',
            features=['linear_features'],
        )

    # Read to Geopandas
    df = gpd.read_file(export_path + f'{airport_id}.linear_features.shp')
    taxi_way = df[
        (df['painted_li'] == 'WIDE_SOLID_YELLOW_WITH_BLACK_BORDER')
        | (df['painted_li'] == 'ILS_CRITICAL_CENTERLINE_WITH_BLACK_BORDER')
    ]

    # Convert dataframe to network
    G = nx.MultiGraph()
    G.graph["crs"] = taxi_way.crs
    key_id = 0
    for line in taxi_way['geometry'].values:
        for seg_start, seg_end in zip(list(line.coords), list(line.coords)[1:]):
            G.add_edge(seg_start, seg_end, key=key_id)
            key_id += 1

    # Add node attributes
    node_attributes = {}
    for i, p in enumerate(G.nodes):
        node_attributes[i] = {'x': p[0], 'y': p[1]}
    G = nx.convert_node_labels_to_integers(G)
    nx.set_node_attributes(G, node_attributes)

    # Add edge attributes
    edge_attributes = {}
    for u, v, key in G.edges:
        length = haversine_distance(G.nodes[u], G.nodes[v])
        edge_attributes[(u, v, key)] = {'length': length}
    nx.set_edge_attributes(G, edge_attributes)

    return G
