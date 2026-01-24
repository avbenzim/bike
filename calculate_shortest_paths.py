"""
Bike Path Shortest Route Calculator for Jerusalem

Calculates shortest paths between area centers using the full bike network:
- Completed bike lanes
- Under construction bike lanes
- Wishing list bike lanes
- Gap connections (auto-generated)

Outputs:
- Distance matrix (CSV)
- Path geometries (GeoJSON)
- Summary statistics (JSON)
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import json
from pathlib import Path
from shapely.geometry import LineString, Point
from scipy.spatial import cKDTree
import fiona
import warnings

warnings.filterwarnings('ignore')

# Configuration
script_dir = Path(__file__).parent
BIKE_LANES_COMPLETED = script_dir / "bike_lanes_completed.kml"
BIKE_LANES_CONSTRUCTION = script_dir / "bike_lanes_construction.kml"
BIKE_LANES_WISHING_LIST = script_dir / "bike_lanes_wishing_list.kml"
AREAS_FILE = script_dir / "jer_areas.shp"
OUTPUT_DIR = script_dir / "output"

# Network parameters
CONNECTION_TOLERANCE = 50  # meters
TARGET_CRS = 2039  # Israel TM Grid

# Enable KML driver
fiona.drvsupport.supported_drivers['KML'] = 'rw'


def load_all_bike_lanes():
    """Load all bike lane files and combine them."""
    print("Loading bike lanes...")

    completed = gpd.read_file(BIKE_LANES_COMPLETED, driver='KML')
    construction = gpd.read_file(BIKE_LANES_CONSTRUCTION, driver='KML')
    wishing_list = gpd.read_file(BIKE_LANES_WISHING_LIST, driver='KML')

    # Add source column
    completed['source'] = 'completed'
    construction['source'] = 'construction'
    wishing_list['source'] = 'wishing_list'

    # Combine all
    all_lanes = gpd.GeoDataFrame(
        pd.concat([completed, construction, wishing_list], ignore_index=True),
        crs=completed.crs
    )

    print(f"  Completed: {len(completed)} segments")
    print(f"  Construction: {len(construction)} segments")
    print(f"  Wishing list: {len(wishing_list)} segments")
    print(f"  Total: {len(all_lanes)} segments")

    return all_lanes


def load_areas():
    """Load Jerusalem areas and calculate centroids."""
    print("Loading areas...")

    areas = gpd.read_file(AREAS_FILE)
    areas = areas[areas['in_jeru'] == 1]

    print(f"  Loaded {len(areas)} Jerusalem areas")

    return areas


def get_endpoints(gdf):
    """Extract start and end points from all linestrings."""
    endpoints = []
    endpoint_info = []

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
            if len(coords) >= 2:
                endpoints.append(Point(coords[0]))
                endpoint_info.append({'lane_idx': idx, 'type': 'start'})
                endpoints.append(Point(coords[-1]))
                endpoint_info.append({'lane_idx': idx, 'type': 'end'})
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                coords = list(line.coords)
                if len(coords) >= 2:
                    endpoints.append(Point(coords[0]))
                    endpoint_info.append({'lane_idx': idx, 'type': 'start'})
                    endpoints.append(Point(coords[-1]))
                    endpoint_info.append({'lane_idx': idx, 'type': 'end'})

    return endpoints, endpoint_info


def connect_network_gaps(bike_lanes, connection_tolerance=50):
    """Connect dangling endpoints to create a more connected network."""
    print(f"Connecting network gaps (tolerance: {connection_tolerance}m)...")

    endpoints, endpoint_info = get_endpoints(bike_lanes)
    n_endpoints = len(endpoints)
    print(f"  Found {n_endpoints} endpoints")

    if n_endpoints == 0:
        return []

    coords = np.array([[p.x, p.y] for p in endpoints])
    tree = cKDTree(coords)

    # Identify dangling endpoints
    TOUCH_TOLERANCE = 1.0
    is_connected = [False] * n_endpoints

    for i in range(n_endpoints):
        nearby = tree.query_ball_point(coords[i], TOUCH_TOLERANCE)
        for j in nearby:
            if i != j and endpoint_info[i]['lane_idx'] != endpoint_info[j]['lane_idx']:
                is_connected[i] = True
                break

    n_dangling = sum(1 for c in is_connected if not c)
    print(f"  Found {n_dangling} dangling endpoints")

    # Connect dangling endpoints
    connections = []
    already_connected = set()

    for i in range(n_endpoints):
        if is_connected[i] or i in already_connected:
            continue

        distances, indices = tree.query(coords[i], k=min(10, n_endpoints))
        best_j, best_dist = None, float('inf')

        for dist, j in zip(distances, indices):
            if j == i or endpoint_info[i]['lane_idx'] == endpoint_info[j]['lane_idx']:
                continue
            if dist > connection_tolerance or dist < 0.1:
                continue

            if not is_connected[j] and j not in already_connected:
                if dist < best_dist:
                    best_dist, best_j = dist, j

        if best_j is None:
            for dist, j in zip(distances, indices):
                if j == i or endpoint_info[i]['lane_idx'] == endpoint_info[j]['lane_idx']:
                    continue
                if dist > connection_tolerance or dist < 0.1:
                    continue
                if dist < best_dist:
                    best_dist, best_j = dist, j
                    break

        if best_j is not None:
            connections.append({
                'geometry': LineString([(endpoints[i].x, endpoints[i].y),
                                        (endpoints[best_j].x, endpoints[best_j].y)]),
                'distance': best_dist
            })
            already_connected.add(i)
            already_connected.add(best_j)

    print(f"  Added {len(connections)} gap connections")
    return connections


def build_network_graph(bike_lanes, connections):
    """Build a NetworkX graph from bike lanes and connections."""
    print("Building network graph...")

    G = nx.Graph()

    # Helper to add a node and return its ID
    node_coords = {}
    node_counter = [0]

    def get_or_create_node(x, y, tolerance=1.0):
        # Check if a node exists nearby
        for node_id, (nx_, ny_) in node_coords.items():
            if abs(x - nx_) < tolerance and abs(y - ny_) < tolerance:
                return node_id

        # Create new node
        node_id = node_counter[0]
        node_counter[0] += 1
        node_coords[node_id] = (x, y)
        G.add_node(node_id, x=x, y=y)
        return node_id

    # Add edges from bike lanes
    for idx, row in bike_lanes.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms)

        for line in lines:
            coords = list(line.coords)
            if len(coords) < 2:
                continue

            # Add edge from start to end
            start_node = get_or_create_node(coords[0][0], coords[0][1])
            end_node = get_or_create_node(coords[-1][0], coords[-1][1])

            if start_node != end_node:
                length = line.length
                if G.has_edge(start_node, end_node):
                    if G[start_node][end_node]['weight'] > length:
                        G[start_node][end_node]['weight'] = length
                else:
                    G.add_edge(start_node, end_node, weight=length)

    # Add edges from gap connections
    for conn in connections:
        coords = list(conn['geometry'].coords)
        start_node = get_or_create_node(coords[0][0], coords[0][1])
        end_node = get_or_create_node(coords[-1][0], coords[-1][1])

        if start_node != end_node:
            length = conn['geometry'].length
            if not G.has_edge(start_node, end_node):
                G.add_edge(start_node, end_node, weight=length)

    print(f"  Network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Analyze connectivity
    components = list(nx.connected_components(G))
    print(f"  Connected components: {len(components)}")
    print(f"  Largest component: {max(len(c) for c in components)} nodes")

    return G, node_coords


def find_nearest_node(point, node_coords):
    """Find the nearest network node to a point."""
    min_dist = float('inf')
    nearest_node = None

    px, py = point.x, point.y

    for node_id, (nx, ny) in node_coords.items():
        dist = ((px - nx)**2 + (py - ny)**2)**0.5
        if dist < min_dist:
            min_dist = dist
            nearest_node = node_id

    return nearest_node, min_dist


def calculate_shortest_paths(G, node_coords, areas):
    """Calculate shortest paths between all pairs of area centers."""
    print("Calculating shortest paths between area centers...")

    # Get area centroids
    centroids = areas.geometry.centroid

    # Get area names
    if 'STAT11_HEB' in areas.columns:
        area_names = areas['STAT11_HEB'].tolist()
    else:
        area_names = [f"Area_{i}" for i in range(len(areas))]

    n_areas = len(areas)
    print(f"  Processing {n_areas} areas...")

    # Find nearest network node for each centroid
    center_nodes = []
    center_distances = []

    for i, centroid in enumerate(centroids):
        node, dist = find_nearest_node(centroid, node_coords)
        center_nodes.append(node)
        center_distances.append(dist)

    avg_dist = np.mean(center_distances)
    print(f"  Average distance from center to network: {avg_dist:.1f}m")

    # Initialize distance matrix
    distance_matrix = np.full((n_areas, n_areas), np.inf)
    np.fill_diagonal(distance_matrix, 0)

    # Store paths
    paths = {}

    # Calculate shortest paths
    total_pairs = n_areas * (n_areas - 1) // 2
    calculated = 0
    connected = 0

    for i in range(n_areas):
        for j in range(i + 1, n_areas):
            calculated += 1

            node_i = center_nodes[i]
            node_j = center_nodes[j]

            if node_i is None or node_j is None:
                continue

            try:
                # Check if nodes are in the same component
                if nx.has_path(G, node_i, node_j):
                    path_length = nx.shortest_path_length(G, node_i, node_j, weight='weight')
                    path_nodes = nx.shortest_path(G, node_i, node_j, weight='weight')

                    # Add distance from centroids to network nodes
                    total_length = path_length + center_distances[i] + center_distances[j]

                    distance_matrix[i, j] = total_length
                    distance_matrix[j, i] = total_length

                    paths[(area_names[i], area_names[j])] = {
                        'nodes': path_nodes,
                        'length': total_length,
                        'network_length': path_length
                    }

                    connected += 1
            except nx.NetworkXNoPath:
                pass

            if calculated % 1000 == 0:
                print(f"    Processed {calculated}/{total_pairs} pairs...")

    print(f"  Connected pairs: {connected}/{total_pairs}")

    # Create DataFrame
    distance_df = pd.DataFrame(
        distance_matrix,
        index=area_names,
        columns=area_names
    )

    return distance_df, paths, area_names


def get_path_geometry(G, node_coords, path_nodes):
    """Reconstruct path geometry from node sequence."""
    if len(path_nodes) < 2:
        return None

    coords = [(node_coords[n][0], node_coords[n][1]) for n in path_nodes]
    return LineString(coords)


def export_results(distance_df, paths, G, node_coords, areas, output_dir):
    """Export results to files."""
    print(f"Exporting results to {output_dir}...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Distance matrix
    distance_df.to_csv(output_dir / "distance_matrix.csv")
    print(f"  Saved distance_matrix.csv")

    # 2. Paths as GeoJSON
    path_features = []
    for (origin, dest), path_data in paths.items():
        geom = get_path_geometry(G, node_coords, path_data['nodes'])
        if geom:
            path_features.append({
                'type': 'Feature',
                'properties': {
                    'origin': origin,
                    'destination': dest,
                    'length_meters': round(path_data['length'], 2),
                    'network_length_meters': round(path_data['network_length'], 2)
                },
                'geometry': geom.__geo_interface__
            })

    if path_features:
        # Transform paths back to WGS84
        paths_gdf = gpd.GeoDataFrame.from_features(path_features, crs=TARGET_CRS)
        paths_gdf = paths_gdf.to_crs(4326)
        paths_gdf.to_file(output_dir / "shortest_paths.geojson", driver='GeoJSON')
        print(f"  Saved shortest_paths.geojson ({len(path_features)} paths)")

    # 3. Area centers
    centers_gdf = areas.copy()
    centers_gdf['geometry'] = areas.geometry.centroid
    centers_gdf = centers_gdf.to_crs(4326)
    centers_gdf.to_file(output_dir / "area_centers.geojson", driver='GeoJSON')
    print(f"  Saved area_centers.geojson")

    # 4. Summary statistics
    valid_distances = distance_df.values[(distance_df.values > 0) & (distance_df.values < np.inf)]

    summary = {
        'total_areas': len(distance_df),
        'total_pairs': len(distance_df) * (len(distance_df) - 1) // 2,
        'connected_pairs': len(valid_distances) // 2,
        'connectivity_pct': round(100 * len(valid_distances) / (len(distance_df) * (len(distance_df) - 1)), 1),
        'avg_path_length_m': round(float(np.mean(valid_distances)), 2) if len(valid_distances) > 0 else None,
        'min_path_length_m': round(float(np.min(valid_distances)), 2) if len(valid_distances) > 0 else None,
        'max_path_length_m': round(float(np.max(valid_distances)), 2) if len(valid_distances) > 0 else None,
        'median_path_length_m': round(float(np.median(valid_distances)), 2) if len(valid_distances) > 0 else None
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary.json")

    return summary


def main():
    print("=" * 70)
    print("BIKE PATH SHORTEST ROUTE CALCULATOR - JERUSALEM")
    print("=" * 70)

    # Load data
    bike_lanes = load_all_bike_lanes()
    areas = load_areas()

    # Transform to projected CRS
    print(f"\nTransforming to CRS {TARGET_CRS} (Israel TM Grid)...")
    bike_lanes = bike_lanes.to_crs(TARGET_CRS)
    areas = areas.to_crs(TARGET_CRS)

    # Connect gaps
    print("\n" + "-" * 50)
    connections = connect_network_gaps(bike_lanes, CONNECTION_TOLERANCE)

    # Build network graph
    print("\n" + "-" * 50)
    G, node_coords = build_network_graph(bike_lanes, connections)

    # Calculate shortest paths
    print("\n" + "-" * 50)
    distance_df, paths, area_names = calculate_shortest_paths(G, node_coords, areas)

    # Export results
    print("\n" + "-" * 50)
    summary = export_results(distance_df, paths, G, node_coords, areas, OUTPUT_DIR)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total areas: {summary['total_areas']}")
    print(f"Total pairs: {summary['total_pairs']}")
    print(f"Connected pairs: {summary['connected_pairs']} ({summary['connectivity_pct']}%)")

    if summary['avg_path_length_m']:
        print(f"\nPath lengths (meters):")
        print(f"  Average: {summary['avg_path_length_m']:,.0f}")
        print(f"  Minimum: {summary['min_path_length_m']:,.0f}")
        print(f"  Maximum: {summary['max_path_length_m']:,.0f}")
        print(f"  Median:  {summary['median_path_length_m']:,.0f}")

    print(f"\nDistance Matrix (first 5x5):")
    print(distance_df.iloc[:5, :5].round(0).to_string())

    print(f"\nResults saved to: {OUTPUT_DIR}")

    return distance_df, paths


if __name__ == '__main__':
    distance_matrix, paths = main()
