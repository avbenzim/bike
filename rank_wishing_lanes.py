"""
Rank wishing list lanes by their individual improvement to total accessibility.
For each lane: compare N(baseline + that lane) vs N(baseline without any wishing list)
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from scipy.spatial import cKDTree
import fiona
import warnings

warnings.filterwarnings('ignore')

# Configuration
script_dir = Path(__file__).parent
ROADS_FILE = script_dir / "jerusalem_roads.kml"
BIKE_LANES_COMPLETED = script_dir / "bike_lanes_completed.kml"
BIKE_LANES_CONSTRUCTION = script_dir / "bike_lanes_construction.kml"
BIKE_LANES_WISHING_LIST = script_dir / "bike_lanes_wishing_list.kml"
AREAS_FILE = script_dir / "jer_areas.shp"

TARGET_CRS = 2039
NODE_TOLERANCE = 15
K_PENALTY = 10
THETA = -1

fiona.drvsupport.supported_drivers['KML'] = 'rw'


def load_data():
    """Load all required data."""
    print("Loading data...")

    areas = gpd.read_file(AREAS_FILE)
    areas = areas[areas['in_jeru'] == 1].copy()
    areas['pop'] = areas['pop_2025'].fillna(0)
    areas['emp'] = areas['emp_2025'].fillna(0)
    areas = areas.to_crs(TARGET_CRS)

    roads = gpd.read_file(ROADS_FILE, driver='KML').to_crs(TARGET_CRS)
    jeru_bounds = areas.total_bounds
    buffer = 1000
    roads = roads.cx[jeru_bounds[0]-buffer:jeru_bounds[2]+buffer,
                     jeru_bounds[1]-buffer:jeru_bounds[3]+buffer]

    completed = gpd.read_file(BIKE_LANES_COMPLETED, driver='KML')
    construction = gpd.read_file(BIKE_LANES_CONSTRUCTION, driver='KML')
    wishing_list = gpd.read_file(BIKE_LANES_WISHING_LIST, driver='KML')

    print(f"  Areas: {len(areas)}, Roads: {len(roads)}")
    print(f"  Completed: {len(completed)}, Construction: {len(construction)}, Wishing: {len(wishing_list)}")

    return areas, roads, completed, construction, wishing_list


def build_network(roads_gdf, bike_lanes_list, tolerance=15):
    """Build NetworkX graph from roads with bike lanes marked."""
    G = nx.Graph()
    coord_to_node = {}
    node_coords = {}
    node_counter = [0]

    def get_or_create_node(x, y):
        key = (round(x / tolerance) * tolerance, round(y / tolerance) * tolerance)
        if key in coord_to_node:
            return coord_to_node[key]
        nid = node_counter[0]
        node_counter[0] += 1
        coord_to_node[key] = nid
        node_coords[nid] = (x, y)
        G.add_node(nid, x=x, y=y)
        return nid

    # Add road edges
    for idx, row in roads_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms) if hasattr(geom, 'geoms') else []
        for line in lines:
            coords = list(line.coords)
            if len(coords) < 2:
                continue
            start = get_or_create_node(coords[0][0], coords[0][1])
            end = get_or_create_node(coords[-1][0], coords[-1][1])
            if start != end:
                length = line.length
                if not G.has_edge(start, end) or G[start][end]['length'] > length:
                    G.add_edge(start, end, length=length, has_bike_lane=False)

    # Add bike lane edges
    for bl_gdf in bike_lanes_list:
        if bl_gdf is None:
            continue
        bl_proj = bl_gdf.to_crs(TARGET_CRS)
        for idx, row in bl_proj.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms) if hasattr(geom, 'geoms') else []
            for line in lines:
                coords = list(line.coords)
                if len(coords) < 2:
                    continue
                start = get_or_create_node(coords[0][0], coords[0][1])
                end = get_or_create_node(coords[-1][0], coords[-1][1])
                if start != end:
                    length = line.length
                    if G.has_edge(start, end):
                        G[start][end]['has_bike_lane'] = True
                    else:
                        G.add_edge(start, end, length=length, has_bike_lane=True)

    # Build node lookup tree
    node_ids = list(node_coords.keys())
    coords_array = np.array([node_coords[n] for n in node_ids])
    node_tree = cKDTree(coords_array)

    return G, node_coords, node_tree, node_ids


def apply_weights(G, k_penalty):
    """Apply K penalty to roads without bike lanes."""
    G = G.copy()
    for u, v in G.edges():
        length = G[u][v]['length']
        has_bike = G[u][v].get('has_bike_lane', False)
        G[u][v]['weight'] = length if has_bike else length * k_penalty
    return G


def calculate_total_N(G, node_coords, node_tree, node_ids, areas_gdf, theta, k_penalty):
    """Calculate total accessibility N."""
    G = apply_weights(G, k_penalty)

    centroids = areas_gdf.geometry.centroid
    n_areas = len(areas_gdf)

    # Find nearest node for each centroid
    center_nodes = []
    for centroid in centroids:
        _, idx = node_tree.query([centroid.x, centroid.y])
        center_nodes.append(node_ids[idx])

    pop = areas_gdf['pop'].values
    emp = areas_gdf['emp'].values

    # Get largest connected component
    largest_cc = max(nx.connected_components(G), key=len)

    total_N = 0
    for i in range(n_areas):
        if center_nodes[i] not in largest_cc:
            continue

        try:
            distances = nx.single_source_dijkstra_path_length(G, center_nodes[i], weight='weight')
        except:
            continue

        for j in range(n_areas):
            if i == j:
                continue
            if center_nodes[j] in distances:
                dist = distances[center_nodes[j]]
                tau = max(dist / 1000, 0.1)
                N_ij = pop[i] * emp[j] * (tau ** theta)
                total_N += N_ij

    return total_N


def main():
    print("=" * 70)
    print("RANKING WISHING LIST LANES BY ACCESSIBILITY IMPROVEMENT")
    print(f"K = {K_PENALTY}, theta = {THETA}")
    print("=" * 70)

    # Load data
    areas, roads, completed, construction, wishing_list = load_data()

    # Calculate baseline (no wishing list lanes)
    print("\nCalculating baseline (without any wishing list lanes)...")
    G_base, node_coords, node_tree, node_ids = build_network(
        roads, [completed, construction], tolerance=NODE_TOLERANCE
    )
    baseline_N = calculate_total_N(G_base, node_coords, node_tree, node_ids, areas, THETA, K_PENALTY)
    print(f"  Baseline N = {baseline_N:.4e}")

    # Evaluate each lane
    print(f"\nEvaluating {len(wishing_list)} wishing list lanes...")
    results = []

    for i in range(len(wishing_list)):
        lane = wishing_list.iloc[[i]]
        lane_name = lane['Name'].iloc[0] if 'Name' in lane.columns else f"Lane_{i}"
        lane_length = lane.to_crs(TARGET_CRS).geometry.length.iloc[0]

        # Build network with this one lane added
        G_with_lane, nc, nt, ni = build_network(
            roads, [completed, construction, lane], tolerance=NODE_TOLERANCE
        )
        N_with_lane = calculate_total_N(G_with_lane, nc, nt, ni, areas, THETA, K_PENALTY)

        improvement = N_with_lane - baseline_N
        pct_improvement = 100 * improvement / baseline_N if baseline_N > 0 else 0
        improvement_per_km = improvement / (lane_length / 1000) if lane_length > 0 else 0

        results.append({
            'rank': 0,
            'name': lane_name,
            'length_m': lane_length,
            'N_with_lane': N_with_lane,
            'improvement': improvement,
            'pct_improvement': pct_improvement,
            'improvement_per_km': improvement_per_km
        })

        print(f"  [{i+1:2d}/{len(wishing_list)}] {lane_name[:40]:40s} +{pct_improvement:.3f}%")

    # Sort by improvement
    results.sort(key=lambda x: x['improvement'], reverse=True)
    for i, r in enumerate(results):
        r['rank'] = i + 1

    # Print ranking
    print("\n" + "=" * 70)
    print("RANKING (by total improvement)")
    print("=" * 70)
    print(f"{'Rank':<5} {'Lane Name':<45} {'Length':>8} {'Improvement':>12} {'%':>8}")
    print("-" * 80)

    for r in results:
        print(f"{r['rank']:<5} {r['name'][:44]:<45} {r['length_m']:>7.0f}m {r['improvement']:>12.2e} {r['pct_improvement']:>7.3f}%")

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(script_dir / 'wishing_list_ranking.csv', index=False)
    print(f"\nSaved ranking to wishing_list_ranking.csv")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Baseline N: {baseline_N:.4e}")
    print(f"\nTop 5 lanes:")
    for r in results[:5]:
        print(f"  {r['rank']}. {r['name'][:50]} (+{r['pct_improvement']:.3f}%)")

    total_all = sum(r['improvement'] for r in results)
    print(f"\nSum of individual improvements: {total_all:.2e} ({100*total_all/baseline_N:.2f}%)")
    print("(Note: actual combined improvement may differ due to network effects)")


if __name__ == '__main__':
    main()
