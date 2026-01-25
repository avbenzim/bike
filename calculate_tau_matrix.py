"""
Calculate distance matrix (tau_ij) between area centers
using full road network + bike lanes

Roads from ISR.parquet, bike lanes from KML files
K penalty applied to roads without bike lanes
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
OUTPUT_DIR = script_dir / "output_with_roads"

TARGET_CRS = 2039  # Israel TM Grid (meters)
K_PENALTY = 10     # Roads without bike lanes are K times slower

fiona.drvsupport.supported_drivers['KML'] = 'rw'


def load_roads(roads_file, bounds, buffer=1000):
    """Load road network from parquet and filter to area."""
    print("Loading road network...")

    df = pd.read_parquet(roads_file)
    print(f"  Total roads in Israel: {len(df)}")

    # Convert WKB geometry
    df['geometry'] = df['geometry'].apply(lambda x: wkb.loads(x))
    roads_gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=4326)
    roads_gdf = roads_gdf.to_crs(TARGET_CRS)

    # Filter to bounds
    roads_gdf = roads_gdf.cx[bounds[0]-buffer:bounds[2]+buffer,
                             bounds[1]-buffer:bounds[3]+buffer]

    print(f"  Roads in area: {len(roads_gdf)}")
    return roads_gdf


def build_network(roads_gdf, node_tolerance=5):
    """Build network graph from roads, linking nearby endpoints."""
    print("Building road network graph...")

    G = nx.Graph()
    all_coords = []
    edge_data = []

    # Extract all line endpoints
    for idx, row in roads_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
            if len(coords) >= 2:
                all_coords.append(coords[0][:2])
                all_coords.append(coords[-1][:2])
                edge_data.append({
                    'start_idx': len(all_coords) - 2,
                    'end_idx': len(all_coords) - 1,
                    'length': geom.length
                })

    if not all_coords:
        return G, {}

    # Cluster nearby nodes
    coords_array = np.array(all_coords)
    tree = cKDTree(coords_array)

    # Map each coordinate to a node ID
    node_map = {}
    node_coords = {}
    node_counter = 0

    for i, coord in enumerate(all_coords):
        if i in node_map:
            continue

        # Find all points within tolerance
        nearby = tree.query_ball_point(coord, node_tolerance)

        # Assign same node ID to all nearby points
        for j in nearby:
            if j not in node_map:
                node_map[j] = node_counter

        node_coords[node_counter] = coord
        G.add_node(node_counter, x=coord[0], y=coord[1])
        node_counter += 1

    # Add edges
    for edge in edge_data:
        start_node = node_map[edge['start_idx']]
        end_node = node_map[edge['end_idx']]

        if start_node != end_node:
            length = edge['length']
            if G.has_edge(start_node, end_node):
                if G[start_node][end_node]['length'] > length:
                    G[start_node][end_node]['length'] = length
            else:
                G.add_edge(start_node, end_node, length=length, has_bike_lane=False)

    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    # Check connectivity
    components = list(nx.connected_components(G))
    print(f"  Connected components: {len(components)}")
    print(f"  Largest component: {max(len(c) for c in components)} nodes")

    return G, node_coords


def add_bike_lanes(G, node_coords, bike_lanes_gdf):
    """Mark edges with bike lanes."""
    print("Adding bike lanes to network...")

    if len(bike_lanes_gdf) == 0:
        return G

    bike_lanes = bike_lanes_gdf.to_crs(TARGET_CRS)

    # Build spatial index for nodes
    node_ids = list(node_coords.keys())
    coords_array = np.array([node_coords[n] for n in node_ids])
    tree = cKDTree(coords_array)

    updated = 0
    added = 0

    for idx, row in bike_lanes.iterrows():
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

            if start_dist < 150 and end_dist < 150:
                if G.has_edge(start_node, end_node):
                    G[start_node][end_node]['has_bike_lane'] = True
                    updated += 1
                else:
                    G.add_edge(start_node, end_node, length=length, has_bike_lane=True)
                    added += 1

    print(f"  Updated {updated} edges, added {added} new edges")
    return G


def apply_weights(G, k_penalty):
    """Apply K penalty to roads without bike lanes."""
    print(f"Applying K={k_penalty} penalty...")

    bike_lane_edges = 0
    road_edges = 0

    for u, v in G.edges():
        length = G[u][v]['length']
        has_bike = G[u][v].get('has_bike_lane', False)

        if has_bike:
            G[u][v]['weight'] = length
            bike_lane_edges += 1
        else:
            G[u][v]['weight'] = length * k_penalty
            road_edges += 1

    print(f"  Bike lane edges: {bike_lane_edges}")
    print(f"  Road edges (with penalty): {road_edges}")
    return G


def calculate_distance_matrix(G, node_coords, areas_gdf):
    """Calculate shortest path distances between area centroids."""
    print("Calculating distance matrix...")

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

    # Calculate distances
    dist_matrix = np.full((n_areas, n_areas), np.inf)
    np.fill_diagonal(dist_matrix, 0)

    connected = 0
    total = n_areas * (n_areas - 1) // 2

    for i in range(n_areas):
        # Use single-source shortest paths for efficiency
        if center_nodes[i] in G:
            lengths = nx.single_source_dijkstra_path_length(G, center_nodes[i], weight='weight')

            for j in range(i + 1, n_areas):
                if center_nodes[j] in lengths:
                    dist_matrix[i, j] = lengths[center_nodes[j]]
                    dist_matrix[j, i] = lengths[center_nodes[j]]
                    connected += 1

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{n_areas} areas...")

    print(f"  Connected pairs: {connected}/{total} ({100*connected/total:.1f}%)")

    return dist_matrix


def main():
    print("=" * 60)
    print("DISTANCE MATRIX CALCULATION (tau_ij)")
    print(f"K = {K_PENALTY}")
    print("=" * 60)

    # Load areas
    print("\nLoading areas...")
    areas = gpd.read_file(AREAS_FILE)
    areas = areas[areas['in_jeru'] == 1].copy()
    areas = areas.to_crs(TARGET_CRS)
    print(f"  Areas: {len(areas)}")

    # Get area names
    if 'name' in areas.columns:
        area_names = areas['name'].fillna('').astype(str).tolist()
    else:
        area_names = [f"Area_{i}" for i in range(len(areas))]

    # Load roads
    bounds = areas.total_bounds
    roads = load_roads(ROADS_FILE, bounds)

    # Build network
    print("\n" + "-" * 40)
    G, node_coords = build_network(roads, node_tolerance=10)

    # Load and add bike lanes
    print("\n" + "-" * 40)
    completed = gpd.read_file(BIKE_LANES_COMPLETED, driver='KML')
    construction = gpd.read_file(BIKE_LANES_CONSTRUCTION, driver='KML')
    wishing = gpd.read_file(BIKE_LANES_WISHING_LIST, driver='KML')

    all_bikes = pd.concat([completed, construction, wishing], ignore_index=True)
    all_bikes = gpd.GeoDataFrame(all_bikes, crs=completed.crs)
    print(f"  Total bike lanes: {len(all_bikes)}")

    G = add_bike_lanes(G, node_coords, all_bikes)

    # Apply weights
    print("\n" + "-" * 40)
    G = apply_weights(G, K_PENALTY)

    # Calculate distance matrix
    print("\n" + "-" * 40)
    dist_matrix = calculate_distance_matrix(G, node_coords, areas)

    # Create output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save distance matrix
    dist_df = pd.DataFrame(dist_matrix, index=area_names, columns=area_names)
    dist_df.to_csv(OUTPUT_DIR / "tau_matrix.csv")
    print(f"\nSaved tau_matrix.csv")

    # Summary stats
    valid = dist_matrix[(dist_matrix > 0) & (dist_matrix < np.inf)]

    summary = {
        'n_areas': len(areas),
        'connected_pairs': len(valid) // 2,
        'total_pairs': len(areas) * (len(areas) - 1) // 2,
        'connectivity_pct': round(100 * len(valid) / (len(areas) * (len(areas) - 1)), 1),
        'mean_distance_m': round(float(np.mean(valid)), 0),
        'median_distance_m': round(float(np.median(valid)), 0),
        'min_distance_m': round(float(np.min(valid)), 0),
        'max_distance_m': round(float(np.max(valid)), 0)
    }

    with open(OUTPUT_DIR / "tau_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Areas: {summary['n_areas']}")
    print(f"Connected pairs: {summary['connected_pairs']}/{summary['total_pairs']} ({summary['connectivity_pct']}%)")
    print(f"Mean distance: {summary['mean_distance_m']:,.0f} m")
    print(f"Median distance: {summary['median_distance_m']:,.0f} m")
    print(f"Range: {summary['min_distance_m']:,.0f} - {summary['max_distance_m']:,.0f} m")

    print("\nDistance Matrix (first 5x5, in meters):")
    print(dist_df.iloc[:5, :5].round(0).to_string())

    return dist_df


if __name__ == '__main__':
    dist_matrix = main()
