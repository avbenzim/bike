"""
Bike Lane Impact Analysis using Gravity Model

This script:
1. Loads road network from ISR.parquet
2. Combines with bike lanes (completed, construction, wishing list)
3. Calculates shortest paths with K penalty for roads without bike lanes
4. Computes gravity model: N_ij = pop_i * emp_j * tau_ij^theta
5. Ranks wishing list lane combinations by total accessibility improvement

Parameters:
- K: Penalty factor for roads without bike lanes (default: 10)
- theta: Distance decay parameter (default: -6)
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import json
from pathlib import Path
from shapely.geometry import LineString, Point
from shapely import wkb
from scipy.spatial import cKDTree
from itertools import combinations
import fiona
import warnings

warnings.filterwarnings('ignore')

# Configuration
script_dir = Path(__file__).parent
ROADS_FILE = script_dir / "ISR.parquet"
BIKE_LANES_COMPLETED = script_dir / "bike_lanes_completed.kml"
BIKE_LANES_CONSTRUCTION = script_dir / "bike_lanes_construction.kml"
BIKE_LANES_WISHING_LIST = script_dir / "bike_lanes_wishing_list.kml"
AREAS_FILE = script_dir / "jer_areas.shp"
OUTPUT_DIR = script_dir / "output_gravity_model"

# Network parameters
TARGET_CRS = 2039  # Israel TM Grid (meters)

# Model parameters
K_PENALTY = 10      # Roads without bike lanes are K times slower
THETA = -1          # Distance decay parameter (negative = decay)

# Enable KML driver
fiona.drvsupport.supported_drivers['KML'] = 'rw'


def load_roads(roads_file, areas_gdf):
    """Load road network from parquet and filter to Jerusalem area."""
    print("Loading road network...")

    df = pd.read_parquet(roads_file)
    print(f"  Total roads in Israel: {len(df)}")

    # Convert WKB geometry to shapely
    df['geometry'] = df['geometry'].apply(lambda x: wkb.loads(x))

    # Create GeoDataFrame (roads are in WGS84)
    roads_gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=4326)

    # Transform to projected CRS
    roads_gdf = roads_gdf.to_crs(TARGET_CRS)

    # Get Jerusalem bounding box with buffer
    jeru_bounds = areas_gdf.total_bounds  # minx, miny, maxx, maxy
    buffer = 1000  # 1km buffer

    # Filter roads to Jerusalem area
    roads_gdf = roads_gdf.cx[jeru_bounds[0]-buffer:jeru_bounds[2]+buffer,
                             jeru_bounds[1]-buffer:jeru_bounds[3]+buffer]

    print(f"  Roads in Jerusalem area: {len(roads_gdf)}")

    return roads_gdf


def load_bike_lanes():
    """Load all bike lane files."""
    print("Loading bike lanes...")

    completed = gpd.read_file(BIKE_LANES_COMPLETED, driver='KML')
    construction = gpd.read_file(BIKE_LANES_CONSTRUCTION, driver='KML')
    wishing_list = gpd.read_file(BIKE_LANES_WISHING_LIST, driver='KML')

    completed['source'] = 'completed'
    completed['is_wishing'] = False
    construction['source'] = 'construction'
    construction['is_wishing'] = False
    wishing_list['source'] = 'wishing_list'
    wishing_list['is_wishing'] = True
    wishing_list['wishing_id'] = range(len(wishing_list))

    print(f"  Completed: {len(completed)}")
    print(f"  Construction: {len(construction)}")
    print(f"  Wishing list: {len(wishing_list)}")

    return completed, construction, wishing_list


def load_areas():
    """Load Jerusalem areas with population and employment data."""
    print("Loading areas...")

    areas = gpd.read_file(AREAS_FILE)
    areas = areas[areas['in_jeru'] == 1].copy()

    # Ensure we have the required columns
    areas['pop'] = areas['pop_2025'].fillna(0)
    areas['emp'] = areas['emp_2025'].fillna(0)

    # Use area name or ID
    if 'name' in areas.columns:
        areas['area_name'] = areas['name'].fillna('').astype(str)
        areas.loc[areas['area_name'] == '', 'area_name'] = areas.index[areas['area_name'] == ''].astype(str)
    else:
        areas['area_name'] = areas.index.astype(str)

    print(f"  Areas: {len(areas)}")
    print(f"  Total population: {areas['pop'].sum():,.0f}")
    print(f"  Total employment: {areas['emp'].sum():,.0f}")

    return areas


def build_road_network(roads_gdf, tolerance=10):
    """Build a NetworkX graph from road geometries."""
    print("Building road network graph...")

    G = nx.Graph()
    node_coords = {}
    node_counter = [0]

    def get_or_create_node(x, y):
        # Check for existing nearby node
        for nid, (nx_, ny_) in node_coords.items():
            if abs(x - nx_) < tolerance and abs(y - ny_) < tolerance:
                return nid

        nid = node_counter[0]
        node_counter[0] += 1
        node_coords[nid] = (x, y)
        G.add_node(nid, x=x, y=y)
        return nid

    for idx, row in roads_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
            if len(coords) >= 2:
                start = get_or_create_node(coords[0][0], coords[0][1])
                end = get_or_create_node(coords[-1][0], coords[-1][1])

                if start != end:
                    length = geom.length
                    if not G.has_edge(start, end) or G[start][end]['length'] > length:
                        G.add_edge(start, end, length=length, has_bike_lane=False)

    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    return G, node_coords


def add_bike_lanes_to_network(G, node_coords, bike_lanes_gdf, k_penalty):
    """Add bike lanes to network, reducing weight where bike lanes exist."""

    if len(bike_lanes_gdf) == 0:
        return G

    bike_lanes_proj = bike_lanes_gdf.to_crs(TARGET_CRS)

    # Build spatial index
    node_ids = list(node_coords.keys())
    coords_array = np.array([node_coords[n] for n in node_ids])

    if len(coords_array) == 0:
        return G

    tree = cKDTree(coords_array)

    updated = 0
    added = 0

    for idx, row in bike_lanes_proj.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms)

        for line in lines:
            coords = list(line.coords)
            if len(coords) < 2:
                continue

            length = line.length

            # Find nearest nodes
            _, start_idx = tree.query(coords[0][:2])
            _, end_idx = tree.query(coords[-1][:2])

            start_node = node_ids[start_idx]
            end_node = node_ids[end_idx]

            if start_node == end_node:
                continue

            # Check distances
            start_dist = np.sqrt((coords[0][0] - coords_array[start_idx][0])**2 +
                                (coords[0][1] - coords_array[start_idx][1])**2)
            end_dist = np.sqrt((coords[-1][0] - coords_array[end_idx][0])**2 +
                              (coords[-1][1] - coords_array[end_idx][1])**2)

            if start_dist < 200 and end_dist < 200:
                if G.has_edge(start_node, end_node):
                    G[start_node][end_node]['has_bike_lane'] = True
                    updated += 1
                else:
                    G.add_edge(start_node, end_node, length=length, has_bike_lane=True)
                    added += 1

    return G


def apply_weights(G, k_penalty):
    """Apply K penalty to roads without bike lanes."""
    for u, v in G.edges():
        length = G[u][v]['length']
        has_bike = G[u][v].get('has_bike_lane', False)
        G[u][v]['weight'] = length if has_bike else length * k_penalty
    return G


def calculate_distance_matrix(G, node_coords, areas_gdf):
    """Calculate shortest path distances between area centroids."""

    centroids = areas_gdf.geometry.centroid
    n_areas = len(areas_gdf)

    # Build spatial index
    node_ids = list(node_coords.keys())
    coords_array = np.array([node_coords[n] for n in node_ids])
    tree = cKDTree(coords_array)

    # Find nearest node for each centroid
    center_nodes = []
    for centroid in centroids:
        _, idx = tree.query([centroid.x, centroid.y])
        center_nodes.append(node_ids[idx])

    # Calculate distance matrix
    dist_matrix = np.full((n_areas, n_areas), np.inf)
    np.fill_diagonal(dist_matrix, 0)

    for i in range(n_areas):
        for j in range(i + 1, n_areas):
            try:
                if nx.has_path(G, center_nodes[i], center_nodes[j]):
                    dist = nx.shortest_path_length(G, center_nodes[i], center_nodes[j], weight='weight')
                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist
            except:
                pass

    return dist_matrix


def calculate_gravity_model(pop, emp, dist_matrix, theta):
    """
    Calculate gravity model accessibility.

    N_ij = pop_i * emp_j * tau_ij^theta

    Returns total sum of N_ij
    """
    n = len(pop)

    # Convert to numpy arrays
    pop = np.array(pop)
    emp = np.array(emp)

    # Calculate tau^theta (avoid division by zero)
    # Use distance in km for more reasonable values
    tau = dist_matrix / 1000  # Convert to km
    tau = np.maximum(tau, 0.1)  # Minimum 100m
    tau_theta = np.power(tau, theta)

    # Set diagonal to 0 (no self-interaction) and inf distances to 0
    tau_theta[dist_matrix == 0] = 0
    tau_theta[np.isinf(dist_matrix)] = 0

    # Calculate N_ij = pop_i * emp_j * tau_ij^theta
    # Using outer product: pop_i * emp_j for all i,j
    pop_emp = np.outer(pop, emp)
    N = pop_emp * tau_theta

    total_N = np.sum(N)

    return total_N, N


def evaluate_scenario(G_base, node_coords, wishing_subset, areas, k_penalty, theta, tree, node_ids, coords_array):
    """Evaluate a scenario with specific wishing list lanes included."""

    # Copy base graph
    G = G_base.copy()

    # Add selected wishing list lanes
    if wishing_subset is not None and len(wishing_subset) > 0:
        wishing_proj = wishing_subset.to_crs(TARGET_CRS)

        for idx, row in wishing_proj.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms)

            for line in lines:
                coords = list(line.coords)
                if len(coords) < 2:
                    continue

                length = line.length

                _, start_idx = tree.query(coords[0][:2])
                _, end_idx = tree.query(coords[-1][:2])

                start_node = node_ids[start_idx]
                end_node = node_ids[end_idx]

                if start_node == end_node:
                    continue

                start_dist = np.sqrt((coords[0][0] - coords_array[start_idx][0])**2 +
                                    (coords[0][1] - coords_array[start_idx][1])**2)
                end_dist = np.sqrt((coords[-1][0] - coords_array[end_idx][0])**2 +
                                  (coords[-1][1] - coords_array[end_idx][1])**2)

                if start_dist < 200 and end_dist < 200:
                    if G.has_edge(start_node, end_node):
                        # Update to bike lane weight (no penalty)
                        G[start_node][end_node]['weight'] = min(G[start_node][end_node]['weight'], length)
                        G[start_node][end_node]['has_bike_lane'] = True
                    else:
                        G.add_edge(start_node, end_node, length=length, weight=length, has_bike_lane=True)

    # Calculate distance matrix
    dist_matrix = calculate_distance_matrix(G, node_coords, areas)

    # Calculate gravity model
    total_N, N_matrix = calculate_gravity_model(
        areas['pop'].values,
        areas['emp'].values,
        dist_matrix,
        theta
    )

    # Calculate connectivity
    connected = np.sum((dist_matrix > 0) & (dist_matrix < np.inf)) / 2

    return {
        'total_N': total_N,
        'connected_pairs': connected,
        'avg_distance': np.mean(dist_matrix[(dist_matrix > 0) & (dist_matrix < np.inf)]) if np.any((dist_matrix > 0) & (dist_matrix < np.inf)) else 0
    }


def rank_wishing_list_lanes(roads_gdf, completed, construction, wishing_list, areas, k_penalty, theta):
    """Rank individual wishing list lanes by their impact on accessibility."""
    print("\n" + "=" * 70)
    print("RANKING WISHING LIST LANES")
    print("=" * 70)

    n_wishing = len(wishing_list)
    print(f"Evaluating {n_wishing} wishing list lanes...")

    # Build base network ONCE
    print("\nBuilding base network...")
    G_base, node_coords = build_road_network(roads_gdf, tolerance=10)
    G_base = add_bike_lanes_to_network(G_base, node_coords, completed, k_penalty)
    G_base = add_bike_lanes_to_network(G_base, node_coords, construction, k_penalty)
    G_base = apply_weights(G_base, k_penalty)

    # Build spatial index
    node_ids = list(node_coords.keys())
    coords_array = np.array([node_coords[n] for n in node_ids])
    tree = cKDTree(coords_array)

    # Calculate baseline (no wishing list)
    print("\nCalculating baseline (without wishing list)...")
    baseline = evaluate_scenario(G_base, node_coords, None, areas, k_penalty, theta, tree, node_ids, coords_array)
    print(f"  Baseline total N: {baseline['total_N']:.2e}")
    print(f"  Connected pairs: {baseline['connected_pairs']:.0f}")

    # Evaluate each lane individually
    print("\nEvaluating individual lanes...")
    lane_impacts = []

    for i in range(n_wishing):
        lane = wishing_list.iloc[[i]]
        result = evaluate_scenario(G_base, node_coords, lane, areas, k_penalty, theta, tree, node_ids, coords_array)

        improvement = result['total_N'] - baseline['total_N']
        pct_improvement = 100 * improvement / baseline['total_N'] if baseline['total_N'] > 0 else 0

        lane_name = lane['Name'].iloc[0] if 'Name' in lane.columns else f"Lane_{i}"

        lane_impacts.append({
            'lane_id': i,
            'name': lane_name,
            'total_N': result['total_N'],
            'improvement': improvement,
            'pct_improvement': pct_improvement,
            'length_m': lane.geometry.length.iloc[0]
        })

        print(f"  Lane {i} ({lane_name[:30]}): {pct_improvement:+.4f}%")

    # Sort by improvement
    lane_impacts.sort(key=lambda x: x['improvement'], reverse=True)

    return baseline, lane_impacts, G_base, node_coords, tree, node_ids, coords_array


def evaluate_combinations(roads_gdf, completed, construction, wishing_list, areas,
                         k_penalty, theta, max_lanes=3):
    """Evaluate combinations of top wishing list lanes."""
    print("\n" + "=" * 70)
    print(f"EVALUATING COMBINATIONS (up to {max_lanes} lanes)")
    print("=" * 70)

    # First rank individual lanes (this also builds the base network)
    baseline, lane_impacts, G_base, node_coords, tree, node_ids, coords_array = rank_wishing_list_lanes(
        roads_gdf, completed, construction, wishing_list, areas, k_penalty, theta
    )

    # Get top lanes
    top_lanes = [l['lane_id'] for l in lane_impacts[:min(8, len(lane_impacts))]]
    print(f"\nEvaluating combinations of top {len(top_lanes)} lanes...")

    combination_results = []

    # Evaluate combinations of 2 and 3 lanes
    for n in range(2, min(max_lanes + 1, len(top_lanes) + 1)):
        print(f"\nCombinations of {n} lanes:")

        for combo in combinations(top_lanes, n):
            lanes_subset = wishing_list.iloc[list(combo)]
            result = evaluate_scenario(G_base, node_coords, lanes_subset, areas, k_penalty, theta, tree, node_ids, coords_array)

            improvement = result['total_N'] - baseline['total_N']
            pct_improvement = 100 * improvement / baseline['total_N'] if baseline['total_N'] > 0 else 0

            total_length = lanes_subset.geometry.length.sum()

            combination_results.append({
                'lanes': combo,
                'n_lanes': n,
                'total_N': result['total_N'],
                'improvement': improvement,
                'pct_improvement': pct_improvement,
                'total_length_m': total_length,
                'improvement_per_km': improvement / (total_length / 1000) if total_length > 0 else 0
            })

            print(f"  {combo}: {pct_improvement:+.4f}%")

    # Sort by improvement
    combination_results.sort(key=lambda x: x['improvement'], reverse=True)

    return baseline, lane_impacts, combination_results


def main():
    print("=" * 70)
    print("BIKE LANE IMPACT ANALYSIS - GRAVITY MODEL")
    print(f"K = {K_PENALTY}, theta = {THETA}")
    print("=" * 70)

    # Load data
    areas = load_areas()
    areas = areas.to_crs(TARGET_CRS)

    roads = load_roads(ROADS_FILE, areas)

    completed, construction, wishing_list = load_bike_lanes()
    completed = completed.to_crs(TARGET_CRS)
    construction = construction.to_crs(TARGET_CRS)
    wishing_list = wishing_list.to_crs(TARGET_CRS)

    # Evaluate combinations
    baseline, lane_impacts, combo_results = evaluate_combinations(
        roads, completed, construction, wishing_list, areas,
        K_PENALTY, THETA, max_lanes=3
    )

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # 1. Lane rankings
    lane_df = pd.DataFrame(lane_impacts)
    lane_df.to_csv(OUTPUT_DIR / "lane_rankings.csv", index=False)
    print(f"  Saved lane_rankings.csv")

    # 2. Combination rankings
    combo_df = pd.DataFrame(combo_results)
    combo_df['lanes'] = combo_df['lanes'].apply(str)
    combo_df.to_csv(OUTPUT_DIR / "combination_rankings.csv", index=False)
    print(f"  Saved combination_rankings.csv")

    # 3. Summary
    summary = {
        'k_penalty': K_PENALTY,
        'theta': THETA,
        'n_areas': len(areas),
        'total_population': float(areas['pop'].sum()),
        'total_employment': float(areas['emp'].sum()),
        'baseline_total_N': float(baseline['total_N']),
        'baseline_connected_pairs': int(baseline['connected_pairs']),
        'n_wishing_list_lanes': len(wishing_list),
        'best_single_lane': lane_impacts[0] if lane_impacts else None,
        'best_combination': combo_results[0] if combo_results else None
    }

    with open(OUTPUT_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved summary.json")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Baseline total N (accessibility): {baseline['total_N']:.4e}")
    print(f"Connected area pairs: {baseline['connected_pairs']:.0f}")

    print("\nTop 5 Individual Lanes:")
    for i, lane in enumerate(lane_impacts[:5]):
        print(f"  {i+1}. Lane {lane['lane_id']} ({lane['name'][:30]}): {lane['pct_improvement']:+.4f}%")

    if combo_results:
        print("\nTop 5 Combinations:")
        for i, combo in enumerate(combo_results[:5]):
            print(f"  {i+1}. Lanes {combo['lanes']}: {combo['pct_improvement']:+.4f}%")

    print(f"\nResults saved to: {OUTPUT_DIR}")

    return baseline, lane_impacts, combo_results


if __name__ == '__main__':
    baseline, lane_impacts, combo_results = main()
