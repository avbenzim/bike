#!/usr/bin/env python3
"""
Test actual network from generate_interactive_map.py
"""
import sys
import os

import geopandas as gpd
import numpy as np
import networkx as nx
from pathlib import Path
from scipy.spatial import cKDTree
from shapely.geometry import Point
from pyproj import Transformer
import fiona
import warnings

warnings.filterwarnings('ignore')

script_dir = Path(__file__).parent
data_dir = script_dir.parent  # Input files and main script are in parent directory
sys.path.insert(0, str(data_dir))  # Add parent to path for imports
fiona.drvsupport.supported_drivers['KML'] = 'rw'
TARGET_CRS = 2039
WGS84 = 4326

# Import the actual load_data and build_network from the main script
from generate_interactive_map import load_data, build_network

def test_actual_network():
    print("=" * 70)
    print("TESTING ACTUAL NETWORK FROM GENERATE_INTERACTIVE_MAP")
    print("=" * 70)

    print("\n1. Loading data...")
    areas, roads, completed, construction, plan, check, wishing = load_data()

    areas_proj = areas.to_crs(TARGET_CRS)
    roads_proj = roads.to_crs(TARGET_CRS)

    print(f"   Areas: {len(areas)}")
    print(f"   Roads: {len(roads)}")

    print("\n2. Building network (using actual build_network)...")
    all_lanes = [completed, construction, plan, check, wishing]
    G, node_coords, node_tree, node_ids, edge_geoms = build_network(
        roads_proj, all_lanes, areas_proj
    )

    print(f"   Nodes: {G.number_of_nodes()}")
    print(f"   Edges: {G.number_of_edges()}")

    # Check connectivity
    print("\n3. Connectivity analysis...")
    components = list(nx.connected_components(G))
    print(f"   Number of components: {len(components)}")

    largest_cc = max(components, key=len)
    print(f"   Largest component: {len(largest_cc)} nodes ({100*len(largest_cc)/G.number_of_nodes():.1f}%)")

    if len(components) > 1:
        print("\n   Other components:")
        transformer = Transformer.from_crs(TARGET_CRS, WGS84, always_xy=True)
        for i, cc in enumerate(sorted(components, key=len, reverse=True)[1:10]):
            nodes_in_cc = list(cc)
            avg_x = np.mean([node_coords[n][0] for n in nodes_in_cc])
            avg_y = np.mean([node_coords[n][1] for n in nodes_in_cc])
            lon, lat = transformer.transform(avg_x, avg_y)
            print(f"     Component {i+2}: {len(cc)} nodes at ({lat:.5f}, {lon:.5f})")

    # Check area connectivity
    print("\n4. Testing area-to-area reachability...")

    # Create weighted graph
    Gw = G.copy()
    K = 100
    for u, v in Gw.edges():
        l = Gw[u][v]['length']
        Gw[u][v]['weight'] = l if Gw[u][v].get('has_bike_lane') else l * K

    # Find area nodes
    areas_proj['area_id'] = range(len(areas_proj))
    area_nodes = {}
    unreachable_areas = []

    for idx, row in areas_proj.iterrows():
        centroid = row.geometry.centroid
        _, nearest_idx = node_tree.query([centroid.x, centroid.y])
        nearest_node = node_ids[nearest_idx]
        area_nodes[row['area_id']] = nearest_node

        if nearest_node not in largest_cc:
            area_name = row.get('name', f'Area {row["area_id"]}')
            unreachable_areas.append((row['area_id'], area_name))

    print(f"   Areas in main component: {len(area_nodes) - len(unreachable_areas)}/{len(area_nodes)}")
    if unreachable_areas:
        print(f"   WARNING: {len(unreachable_areas)} areas NOT in main component!")
        for aid, name in unreachable_areas[:10]:
            print(f"     - Area {aid}: {name}")

    # Test random paths
    print("\n5. Testing random paths...")
    np.random.seed(42)
    n_areas = len(areas_proj)
    success = 0
    failure = 0

    for _ in range(50):
        i, j = np.random.choice(n_areas, 2, replace=False)
        source = area_nodes[i]
        target = area_nodes[j]

        try:
            path_length = nx.dijkstra_path_length(Gw, source, target, weight='weight')
            success += 1
        except nx.NetworkXNoPath:
            failure += 1

    print(f"   Success: {success}/50")
    print(f"   Failure: {failure}/50")

    # Test specific cross-city routes
    print("\n6. Cross-city route tests...")
    transformer = Transformer.from_crs(TARGET_CRS, WGS84, always_xy=True)

    # Find areas at extremes
    area_centers = []
    for idx, row in areas_proj.iterrows():
        centroid = row.geometry.centroid
        area_centers.append({
            'idx': row['area_id'],
            'x': centroid.x,
            'y': centroid.y,
            'name': row.get('name', f'Area {row["area_id"]}')
        })

    sorted_by_x = sorted(area_centers, key=lambda a: a['x'])
    sorted_by_y = sorted(area_centers, key=lambda a: a['y'])

    test_routes = [
        (sorted_by_x[5], sorted_by_x[-5], 'West to East'),
        (sorted_by_y[5], sorted_by_y[-5], 'South to North'),
    ]

    for origin, dest, name in test_routes:
        print(f"\n   {name}:")
        print(f"     From: {origin['name']}")
        print(f"     To: {dest['name']}")

        source = area_nodes[origin['idx']]
        target = area_nodes[dest['idx']]

        try:
            path = nx.dijkstra_path(Gw, source, target, weight='weight')
            path_length = nx.dijkstra_path_length(Gw, source, target, weight='weight')

            # Calculate stats
            actual_dist = 0
            bike_dist = 0
            virtual_dist = 0
            for k in range(len(path) - 1):
                u, v = path[k], path[k+1]
                edge_len = G[u][v]['length']
                actual_dist += edge_len
                if G[u][v].get('has_bike_lane'):
                    bike_dist += edge_len
                if G[u][v].get('is_virtual'):
                    virtual_dist += edge_len

            bike_pct = 100 * bike_dist / actual_dist if actual_dist > 0 else 0

            print(f"     Distance: {actual_dist/1000:.2f} km")
            print(f"     Bike lanes: {bike_pct:.1f}%")
            print(f"     Virtual edges: {100*virtual_dist/actual_dist:.1f}% of route")

        except nx.NetworkXNoPath:
            print("     ERROR: No path found!")

    # Bike lane network analysis
    print("\n7. Bike lane network fragmentation...")
    bike_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('has_bike_lane')]
    G_bike = G.edge_subgraph(bike_edges).copy()

    bike_components = list(nx.connected_components(G_bike))
    print(f"   Bike-only components: {len(bike_components)}")

    if len(bike_components) > 1:
        sorted_comps = sorted(bike_components, key=len, reverse=True)
        print(f"   Largest: {len(sorted_comps[0])} nodes")
        if len(sorted_comps) > 1:
            print(f"   Second: {len(sorted_comps[1])} nodes")

        # Find gaps between top components
        print("\n   Gaps between top bike components:")

        main_comp = sorted_comps[0]
        main_coords = np.array([node_coords[n] for n in main_comp])
        main_tree = cKDTree(main_coords)
        main_nodes = list(main_comp)

        for i, comp in enumerate(sorted_comps[1:5], 2):
            comp_nodes = list(comp)
            min_dist = float('inf')
            for node in comp_nodes:
                coord = node_coords[node]
                dist, _ = main_tree.query([coord[0], coord[1]])
                if dist < min_dist:
                    min_dist = dist

            print(f"     Component {i} ({len(comp)} nodes): {min_dist:.0f}m from main")

    return G, node_coords, area_nodes, areas_proj


if __name__ == "__main__":
    G, node_coords, area_nodes, areas_proj = test_actual_network()

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
