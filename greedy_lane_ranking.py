"""
Greedy Sequential Wishing List Lane Ranking

This script ranks wishing list lanes sequentially:
1. Start with existing network (completed + construction)
2. Find the best wishing list lane to add (biggest improvement)
3. Add it to the network, then find the next best lane
4. Continue until all lanes are ranked

Key feature: Creates intersection nodes where roads meet bike lanes.

Parameters:
- K: Penalty factor for roads without bike lanes (default: 10)
- theta: Distance decay parameter (default: -1)
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
OUTPUT_DIR = script_dir / "output_greedy_ranking"

# Network parameters
TARGET_CRS = 2039  # Israel TM Grid (meters)
NODE_TOLERANCE = 10  # meters - snap nearby nodes together

# Model parameters
K_PENALTY = 10      # Roads without bike lanes are K times slower
THETA = -1          # Distance decay parameter

# Enable KML driver
fiona.drvsupport.supported_drivers['KML'] = 'rw'


def load_roads(roads_file, areas_gdf):
    """Load road network from parquet and filter to Jerusalem area."""
    print("Loading road network...")

    df = pd.read_parquet(roads_file)
    print(f"  Total roads in Israel: {len(df)}")

    df['geometry'] = df['geometry'].apply(lambda x: wkb.loads(x))
    roads_gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=4326)
    roads_gdf = roads_gdf.to_crs(TARGET_CRS)

    jeru_bounds = areas_gdf.total_bounds
    buffer = 1000
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

    print(f"  Completed: {len(completed)}")
    print(f"  Construction: {len(construction)}")
    print(f"  Wishing list: {len(wishing_list)}")

    return completed, construction, wishing_list


def load_areas():
    """Load Jerusalem areas with population and employment data."""
    print("Loading areas...")

    areas = gpd.read_file(AREAS_FILE)
    areas = areas[areas['in_jeru'] == 1].copy()
    areas['pop'] = areas['pop_2025'].fillna(0)
    areas['emp'] = areas['emp_2025'].fillna(0)

    print(f"  Areas: {len(areas)}")
    print(f"  Total population: {areas['pop'].sum():,.0f}")
    print(f"  Total employment: {areas['emp'].sum():,.0f}")

    return areas


def find_intersection_points(roads_gdf, bike_lanes_gdf, buffer_dist=5):
    """Find points where bike lanes intersect or are very close to roads."""
    print("  Finding road-bike lane intersections...")

    intersection_points = []

    # Create spatial index for roads
    roads_sindex = roads_gdf.sindex

    for idx, bike_row in bike_lanes_gdf.iterrows():
        bike_geom = bike_row.geometry
        if bike_geom is None or bike_geom.is_empty:
            continue

        # Get potential road intersections
        possible_matches_idx = list(roads_sindex.intersection(bike_geom.buffer(buffer_dist).bounds))

        for road_idx in possible_matches_idx:
            road_geom = roads_gdf.iloc[road_idx].geometry
            if road_geom is None or road_geom.is_empty:
                continue

            # Check for actual intersection
            if bike_geom.intersects(road_geom):
                intersection = bike_geom.intersection(road_geom)
                if intersection.geom_type == 'Point':
                    intersection_points.append((intersection.x, intersection.y))
                elif intersection.geom_type == 'MultiPoint':
                    for pt in intersection.geoms:
                        intersection_points.append((pt.x, pt.y))

            # Also check for near-misses (bike lane endpoints near roads)
            if bike_geom.geom_type == 'LineString':
                for pt in [Point(bike_geom.coords[0]), Point(bike_geom.coords[-1])]:
                    dist = pt.distance(road_geom)
                    if dist < buffer_dist:
                        nearest = road_geom.interpolate(road_geom.project(pt))
                        intersection_points.append((nearest.x, nearest.y))

    print(f"  Found {len(intersection_points)} intersection points")
    return intersection_points


def build_network_with_intersections(roads_gdf, bike_lanes_list, tolerance=NODE_TOLERANCE):
    """Build network ensuring nodes at road-bike lane intersections."""
    print("Building road network with intersection nodes...")

    # Combine all bike lanes
    all_bike_lanes = pd.concat(bike_lanes_list, ignore_index=True)

    # Find intersection points first
    intersection_points = find_intersection_points(roads_gdf, all_bike_lanes)

    # Collect all points: road endpoints + intersection points + bike lane vertices
    all_points = []

    # Road endpoints
    print("  Collecting road endpoints...")
    for idx, row in roads_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or geom.geom_type != 'LineString':
            continue
        coords = list(geom.coords)
        if len(coords) >= 2:
            all_points.append((coords[0][0], coords[0][1], 'road_start', idx))
            all_points.append((coords[-1][0], coords[-1][1], 'road_end', idx))

    # Intersection points
    for i, (x, y) in enumerate(intersection_points):
        all_points.append((x, y, 'intersection', i))

    # Bike lane vertices
    print("  Collecting bike lane vertices...")
    for bike_gdf in bike_lanes_list:
        for idx, row in bike_gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms)
            for line in lines:
                for coord in line.coords:
                    all_points.append((coord[0], coord[1], 'bike', idx))

    print(f"  Total points to cluster: {len(all_points)}")

    # Cluster nearby points into nodes using KD-tree
    coords_array = np.array([(p[0], p[1]) for p in all_points])
    tree = cKDTree(coords_array)

    # Find clusters using tolerance
    visited = set()
    node_coords = {}
    point_to_node = {}
    node_counter = 0

    for i in range(len(all_points)):
        if i in visited:
            continue

        # Find all points within tolerance
        neighbors = tree.query_ball_point(coords_array[i], tolerance)

        # Create a node at the centroid of the cluster
        cluster_coords = coords_array[neighbors]
        centroid = cluster_coords.mean(axis=0)

        node_id = node_counter
        node_counter += 1
        node_coords[node_id] = (centroid[0], centroid[1])

        for j in neighbors:
            visited.add(j)
            point_to_node[j] = node_id

    print(f"  Created {len(node_coords)} nodes")

    # Build graph
    G = nx.Graph()
    for nid, (x, y) in node_coords.items():
        G.add_node(nid, x=x, y=y)

    # Build spatial index of nodes for fast lookup
    node_ids = list(node_coords.keys())
    node_coords_array = np.array([node_coords[n] for n in node_ids])
    node_tree = cKDTree(node_coords_array)

    # Add road edges
    print("  Adding road edges...")
    for idx, row in roads_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or geom.geom_type != 'LineString':
            continue

        coords = list(geom.coords)
        if len(coords) < 2:
            continue

        # Find nearest nodes for endpoints
        _, start_idx = node_tree.query(coords[0][:2])
        _, end_idx = node_tree.query(coords[-1][:2])

        start_node = node_ids[start_idx]
        end_node = node_ids[end_idx]

        if start_node != end_node:
            length = geom.length
            if not G.has_edge(start_node, end_node) or G[start_node][end_node]['length'] > length:
                G.add_edge(start_node, end_node, length=length, has_bike_lane=False)

    print(f"  Road edges: {G.number_of_edges()}")

    # Add bike lane edges
    print("  Adding bike lane edges...")
    bike_edges = 0
    for bike_gdf in bike_lanes_list:
        for idx, row in bike_gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms)

            for line in lines:
                coords = list(line.coords)
                if len(coords) < 2:
                    continue

                # Connect consecutive vertices
                prev_node = None
                for i, coord in enumerate(coords):
                    x, y = coord[0], coord[1]
                    _, node_idx = node_tree.query([x, y])
                    node = node_ids[node_idx]

                    if prev_node is not None and prev_node != node:
                        segment_length = Point(coords[i-1][:2]).distance(Point(x, y))

                        if G.has_edge(prev_node, node):
                            G[prev_node][node]['has_bike_lane'] = True
                            # Keep shorter length
                            G[prev_node][node]['length'] = min(G[prev_node][node]['length'], segment_length)
                        else:
                            G.add_edge(prev_node, node, length=segment_length, has_bike_lane=True)
                        bike_edges += 1

                    prev_node = node

    print(f"  Bike lane edges: {bike_edges}")
    print(f"  Final network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return G, node_coords


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

    node_ids = list(node_coords.keys())
    coords_array = np.array([node_coords[n] for n in node_ids])
    tree = cKDTree(coords_array)

    center_nodes = []
    for centroid in centroids:
        _, idx = tree.query([centroid.x, centroid.y])
        center_nodes.append(node_ids[idx])

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
    """Calculate gravity model accessibility."""

    pop = np.array(pop)
    emp = np.array(emp)

    tau = dist_matrix / 1000  # Convert to km
    tau = np.maximum(tau, 0.1)  # Minimum 100m
    tau_theta = np.power(tau, theta)

    tau_theta[dist_matrix == 0] = 0
    tau_theta[np.isinf(dist_matrix)] = 0

    pop_emp = np.outer(pop, emp)
    N = pop_emp * tau_theta

    return np.sum(N)


def add_wishing_lane_to_network(G, node_coords, lane_gdf, k_penalty):
    """Add a wishing list lane to the network."""

    G = G.copy()

    node_ids = list(node_coords.keys())
    coords_array = np.array([node_coords[n] for n in node_ids])
    tree = cKDTree(coords_array)

    for idx, row in lane_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms)

        for line in lines:
            coords = list(line.coords)
            if len(coords) < 2:
                continue

            # Connect consecutive vertices
            prev_node = None
            for i, coord in enumerate(coords):
                x, y = coord[0], coord[1]
                _, node_idx = tree.query([x, y])
                node = node_ids[node_idx]

                if prev_node is not None and prev_node != node:
                    segment_length = Point(coords[i-1][:2]).distance(Point(x, y))

                    if G.has_edge(prev_node, node):
                        G[prev_node][node]['has_bike_lane'] = True
                        G[prev_node][node]['weight'] = min(G[prev_node][node]['weight'], segment_length)
                    else:
                        G.add_edge(prev_node, node, length=segment_length, weight=segment_length, has_bike_lane=True)

                prev_node = node

    return G


def greedy_rank_lanes(roads_gdf, completed, construction, wishing_list, areas, k_penalty, theta):
    """Greedy sequential ranking of wishing list lanes."""

    print("\n" + "=" * 70)
    print("GREEDY SEQUENTIAL LANE RANKING")
    print(f"K = {k_penalty}, theta = {theta}")
    print("=" * 70)

    # Build base network with intersection nodes
    G, node_coords = build_network_with_intersections(
        roads_gdf,
        [completed, construction]
    )
    G = apply_weights(G, k_penalty)

    # Calculate baseline
    print("\nCalculating baseline...")
    dist_matrix = calculate_distance_matrix(G, node_coords, areas)
    baseline_N = calculate_gravity_model(areas['pop'].values, areas['emp'].values, dist_matrix, theta)

    connected = np.sum((dist_matrix > 0) & (dist_matrix < np.inf)) / 2
    print(f"  Baseline N: {baseline_N:.4e}")
    print(f"  Connected pairs: {connected:.0f}")

    # Track remaining lanes
    remaining_lanes = list(range(len(wishing_list)))

    # Results
    ranking = []
    current_N = baseline_N
    G_current = G.copy()

    print("\n" + "-" * 70)
    print("GREEDY RANKING")
    print("-" * 70)

    rank = 1
    while remaining_lanes:
        print(f"\nRound {rank}: Evaluating {len(remaining_lanes)} remaining lanes...")

        best_lane = None
        best_N = current_N
        best_improvement = 0
        best_G = None

        for lane_id in remaining_lanes:
            lane = wishing_list.iloc[[lane_id]]
            G_test = add_wishing_lane_to_network(G_current, node_coords, lane, k_penalty)

            dist_matrix = calculate_distance_matrix(G_test, node_coords, areas)
            new_N = calculate_gravity_model(areas['pop'].values, areas['emp'].values, dist_matrix, theta)

            improvement = new_N - current_N

            if improvement > best_improvement:
                best_improvement = improvement
                best_N = new_N
                best_lane = lane_id
                best_G = G_test

        if best_lane is None:
            # No improvement from any remaining lane
            for lane_id in remaining_lanes:
                lane_name = wishing_list.iloc[lane_id]['Name'] if 'Name' in wishing_list.columns else f"Lane_{lane_id}"
                ranking.append({
                    'rank': rank,
                    'lane_id': lane_id,
                    'name': lane_name,
                    'N_after': current_N,
                    'improvement': 0,
                    'pct_improvement': 0,
                    'cumulative_N': current_N,
                    'cumulative_pct_improvement': 100 * (current_N - baseline_N) / baseline_N
                })
                rank += 1
            break

        # Add best lane to network
        G_current = best_G

        lane_name = wishing_list.iloc[best_lane]['Name'] if 'Name' in wishing_list.columns else f"Lane_{best_lane}"
        pct_improvement = 100 * best_improvement / current_N if current_N > 0 else 0
        cumulative_pct = 100 * (best_N - baseline_N) / baseline_N if baseline_N > 0 else 0

        ranking.append({
            'rank': rank,
            'lane_id': best_lane,
            'name': lane_name,
            'N_after': best_N,
            'improvement': best_improvement,
            'pct_improvement': pct_improvement,
            'cumulative_N': best_N,
            'cumulative_pct_improvement': cumulative_pct
        })

        print(f"  Rank {rank}: {lane_name}")
        print(f"    Improvement: +{pct_improvement:.4f}%")
        print(f"    Cumulative: +{cumulative_pct:.4f}%")

        remaining_lanes.remove(best_lane)
        current_N = best_N
        rank += 1

    return baseline_N, ranking


def main():
    print("=" * 70)
    print("GREEDY SEQUENTIAL WISHING LIST LANE RANKING")
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

    # Run greedy ranking
    baseline_N, ranking = greedy_rank_lanes(
        roads, completed, construction, wishing_list, areas,
        K_PENALTY, THETA
    )

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # 1. Ranking
    ranking_df = pd.DataFrame(ranking)
    ranking_df.to_csv(OUTPUT_DIR / "greedy_ranking.csv", index=False)
    print(f"  Saved greedy_ranking.csv")

    # 2. Summary
    summary = {
        'k_penalty': K_PENALTY,
        'theta': THETA,
        'n_areas': len(areas),
        'total_population': float(areas['pop'].sum()),
        'total_employment': float(areas['emp'].sum()),
        'baseline_N': float(baseline_N),
        'final_N': float(ranking[-1]['cumulative_N']) if ranking else float(baseline_N),
        'total_improvement_pct': float(ranking[-1]['cumulative_pct_improvement']) if ranking else 0,
        'n_wishing_list_lanes': len(wishing_list),
        'ranking': ranking
    }

    with open(OUTPUT_DIR / "greedy_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
    print(f"  Saved greedy_summary.json")

    # Print summary
    print("\n" + "=" * 70)
    print("GREEDY RANKING RESULTS")
    print("=" * 70)
    print(f"Baseline N: {baseline_N:.4e}")
    print(f"Final N: {ranking[-1]['cumulative_N']:.4e}" if ranking else "No lanes ranked")
    print(f"Total improvement: +{ranking[-1]['cumulative_pct_improvement']:.4f}%" if ranking else "0%")

    print("\nFull Ranking (in order of selection):")
    for r in ranking:
        print(f"  {r['rank']:2d}. {r['name'][:40]:<40} | +{r['pct_improvement']:.4f}% | Cumulative: +{r['cumulative_pct_improvement']:.4f}%")

    print(f"\nResults saved to: {OUTPUT_DIR}")

    return baseline_N, ranking


if __name__ == '__main__':
    baseline_N, ranking = main()
