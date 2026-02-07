"""
Bike Path Shortest Route Calculator for Jerusalem - With Road Network

Calculates shortest paths between area centers using:
- Bike lanes (weight = actual distance)
- Roads without bike lanes (weight = distance * K, where K is penalty factor)

This allows routing through the entire road network, but strongly prefers
bike lanes when available.

Parameters:
- K: Penalty factor for roads without bike lanes (default: 10)
     K=1 means roads are equivalent to bike lanes
     K=10 means roads are 10x slower than bike lanes
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
OUTPUT_DIR = script_dir / "output_with_roads"

# Network parameters
CONNECTION_TOLERANCE = 50  # meters
TARGET_CRS = 2039  # Israel TM Grid

# Road penalty factor
# K=1: roads same as bike lanes
# K=10: roads 10x slower than bike lanes
K_PENALTY = 10

# Enable KML driver
fiona.drvsupport.supported_drivers['KML'] = 'rw'


def load_all_bike_lanes():
    """Load all bike lane files and combine them."""
    print("Loading bike lanes...")

    completed = gpd.read_file(BIKE_LANES_COMPLETED, driver='KML')
    construction = gpd.read_file(BIKE_LANES_CONSTRUCTION, driver='KML')
    wishing_list = gpd.read_file(BIKE_LANES_WISHING_LIST, driver='KML')

    completed['source'] = 'bike_lane'
    construction['source'] = 'bike_lane'
    wishing_list['source'] = 'bike_lane'

    all_lanes = gpd.GeoDataFrame(
        pd.concat([completed, construction, wishing_list], ignore_index=True),
        crs=completed.crs
    )

    print(f"  Completed: {len(completed)} segments")
    print(f"  Construction: {len(construction)} segments")
    print(f"  Wishing list: {len(wishing_list)} segments")
    print(f"  Total bike lanes: {len(all_lanes)} segments")

    return all_lanes


def load_areas():
    """Load Jerusalem areas."""
    print("Loading areas...")
    areas = gpd.read_file(AREAS_FILE)
    areas = areas[areas['in_jeru'] == 1]
    print(f"  Loaded {len(areas)} Jerusalem areas")
    return areas


def create_road_network_from_areas(areas_gdf):
    """
    Create a synthetic road network by connecting adjacent areas.

    This creates a graph where:
    - Each area centroid is a node
    - Adjacent areas (touching/nearby) are connected by edges
    - Edge weight is the distance between centroids
    """
    print("Creating road network from area adjacencies...")

    G = nx.Graph()

    # Get centroids
    centroids = areas_gdf.geometry.centroid
    n_areas = len(areas_gdf)

    # Add nodes at centroids
    for i in range(n_areas):
        G.add_node(i, x=centroids.iloc[i].x, y=centroids.iloc[i].y, type='area_center')

    # Find adjacent areas (buffer and intersect)
    print("  Finding adjacent areas...")
    buffer_dist = 100  # 100m buffer to find nearby areas

    adjacencies = 0
    for i in range(n_areas):
        geom_i = areas_gdf.geometry.iloc[i]
        buffered = geom_i.buffer(buffer_dist)

        for j in range(i + 1, n_areas):
            geom_j = areas_gdf.geometry.iloc[j]

            # Check if areas are adjacent (touching or within buffer)
            if buffered.intersects(geom_j):
                # Calculate distance between centroids
                dist = centroids.iloc[i].distance(centroids.iloc[j])

                G.add_edge(i, j,
                          weight=dist,
                          length=dist,
                          type='road',
                          has_bike_lane=False)
                adjacencies += 1

    print(f"  Found {adjacencies} adjacent area pairs")
    print(f"  Road network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Check connectivity
    components = list(nx.connected_components(G))
    print(f"  Connected components: {len(components)}")

    if len(components) > 1:
        # Connect components by adding edges between closest nodes
        print("  Connecting isolated components...")
        while len(components) > 1:
            # Find two closest nodes from different components
            min_dist = float('inf')
            best_edge = None

            for c1 in range(len(components)):
                for c2 in range(c1 + 1, len(components)):
                    for n1 in components[c1]:
                        for n2 in components[c2]:
                            dist = centroids.iloc[n1].distance(centroids.iloc[n2])
                            if dist < min_dist:
                                min_dist = dist
                                best_edge = (n1, n2, dist)

            if best_edge:
                G.add_edge(best_edge[0], best_edge[1],
                          weight=best_edge[2],
                          length=best_edge[2],
                          type='road',
                          has_bike_lane=False)

            components = list(nx.connected_components(G))

        print(f"  Network is now fully connected!")

    return G


def create_combined_network(bike_lanes_gdf, road_graph, areas_gdf, k_penalty=10):
    """
    Create a combined network graph with bike lanes and roads.

    Bike lanes have weight = actual distance
    Roads without bike lanes have weight = distance * k_penalty
    """
    print(f"Creating combined network (K={k_penalty})...")

    # Start with the road graph (already in projected CRS)
    G = road_graph.copy()

    # Apply K penalty to all road edges
    print("  Applying K penalty to road edges...")
    for u, v in G.edges():
        length = G[u][v]['length']
        G[u][v]['weight'] = length * k_penalty

    # Node coordinate lookup
    node_coords = {n: (G.nodes[n]['x'], G.nodes[n]['y']) for n in G.nodes()}

    print(f"  Road network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Build spatial index for road nodes
    road_node_ids = list(node_coords.keys())
    road_coords = np.array([node_coords[n] for n in road_node_ids])
    road_tree = cKDTree(road_coords)

    # Process bike lanes - add bike lane edges with no penalty
    print("  Processing bike lanes...")
    bike_lanes_proj = bike_lanes_gdf.to_crs(TARGET_CRS)

    bike_lane_edges_updated = 0
    bike_lane_edges_added = 0

    for idx, row in bike_lanes_proj.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms)

        for line in lines:
            coords = list(line.coords)
            if len(coords) < 2:
                continue

            start_coord = coords[0]
            end_coord = coords[-1]
            length = line.length

            # Find nearest road nodes to bike lane endpoints
            _, start_idx = road_tree.query(start_coord)
            _, end_idx = road_tree.query(end_coord)

            start_node = road_node_ids[start_idx]
            end_node = road_node_ids[end_idx]

            if start_node == end_node:
                continue

            # Check distance to nearest road node
            start_dist = ((start_coord[0] - road_coords[start_idx][0])**2 +
                         (start_coord[1] - road_coords[start_idx][1])**2)**0.5
            end_dist = ((end_coord[0] - road_coords[end_idx][0])**2 +
                       (end_coord[1] - road_coords[end_idx][1])**2)**0.5

            # If bike lane endpoints are reasonably close to road nodes
            if start_dist < 300 and end_dist < 300:
                if G.has_edge(start_node, end_node):
                    # Update existing edge - bike lane, no penalty
                    current_weight = G[start_node][end_node]['weight']
                    # Use actual length (no K penalty for bike lanes)
                    new_weight = min(current_weight, length)
                    G[start_node][end_node]['weight'] = new_weight
                    G[start_node][end_node]['has_bike_lane'] = True
                    G[start_node][end_node]['type'] = 'bike_lane'
                    bike_lane_edges_updated += 1
                else:
                    # Add new edge for bike lane (no penalty)
                    G.add_edge(start_node, end_node,
                              weight=length,
                              length=length,
                              type='bike_lane',
                              has_bike_lane=True)
                    bike_lane_edges_added += 1

    print(f"    Updated {bike_lane_edges_updated} road edges with bike lanes")
    print(f"    Added {bike_lane_edges_added} new bike lane edges")

    # Final statistics
    bike_lane_edges = sum(1 for _, _, d in G.edges(data=True) if d.get('has_bike_lane', False))
    road_only_edges = G.number_of_edges() - bike_lane_edges

    print(f"  Combined network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"    Edges with bike lanes: {bike_lane_edges}")
    print(f"    Road-only edges (K={k_penalty} penalty): {road_only_edges}")

    # Check connectivity
    if nx.is_connected(G):
        print("  Network is fully connected!")
    else:
        components = list(nx.connected_components(G))
        print(f"  Network has {len(components)} components")
        print(f"  Largest component: {max(len(c) for c in components)} nodes")

    return G, node_coords


def find_nearest_node(point, node_coords):
    """Find nearest network node to a point."""
    min_dist = float('inf')
    nearest_node = None

    px, py = point.x, point.y

    for node_id, (nx, ny) in node_coords.items():
        dist = ((px - nx)**2 + (py - ny)**2)**0.5
        if dist < min_dist:
            min_dist = dist
            nearest_node = node_id

    return nearest_node, min_dist


def calculate_shortest_paths(G, node_coords, areas, k_penalty):
    """Calculate shortest paths between all area centers."""
    print("Calculating shortest paths...")

    centroids = areas.geometry.centroid

    if 'STAT11_HEB' in areas.columns:
        area_names = areas['STAT11_HEB'].tolist()
    else:
        area_names = [f"Area_{i}" for i in range(len(areas))]

    n_areas = len(areas)
    print(f"  Processing {n_areas} areas...")

    # Build spatial index for faster node lookup
    node_ids = list(node_coords.keys())
    coords_array = np.array([node_coords[n] for n in node_ids])
    tree = cKDTree(coords_array)

    # Find nearest node for each centroid
    center_nodes = []
    center_distances = []

    for centroid in centroids:
        _, idx = tree.query([centroid.x, centroid.y])
        center_nodes.append(node_ids[idx])
        dist = ((centroid.x - coords_array[idx][0])**2 +
                (centroid.y - coords_array[idx][1])**2)**0.5
        center_distances.append(dist)

    avg_dist = np.mean(center_distances)
    print(f"  Average distance from center to network: {avg_dist:.1f}m")

    # Initialize matrices
    distance_matrix = np.full((n_areas, n_areas), np.inf)
    bike_lane_matrix = np.full((n_areas, n_areas), np.inf)  # Distance on bike lanes only
    np.fill_diagonal(distance_matrix, 0)
    np.fill_diagonal(bike_lane_matrix, 0)

    paths = {}

    total_pairs = n_areas * (n_areas - 1) // 2
    calculated = 0
    connected = 0

    for i in range(n_areas):
        for j in range(i + 1, n_areas):
            calculated += 1

            node_i = center_nodes[i]
            node_j = center_nodes[j]

            try:
                if nx.has_path(G, node_i, node_j):
                    # Get shortest path
                    path_nodes = nx.shortest_path(G, node_i, node_j, weight='weight')
                    path_weight = nx.shortest_path_length(G, node_i, node_j, weight='weight')

                    # Calculate actual distance and bike lane distance
                    actual_distance = 0
                    bike_lane_distance = 0

                    for k in range(len(path_nodes) - 1):
                        edge_data = G[path_nodes[k]][path_nodes[k+1]]
                        edge_length = edge_data.get('length', edge_data['weight'] / k_penalty)
                        actual_distance += edge_length

                        if edge_data.get('has_bike_lane', False):
                            bike_lane_distance += edge_length

                    # Add distance from centroids to network
                    total_distance = actual_distance + center_distances[i] + center_distances[j]

                    distance_matrix[i, j] = total_distance
                    distance_matrix[j, i] = total_distance
                    bike_lane_matrix[i, j] = bike_lane_distance
                    bike_lane_matrix[j, i] = bike_lane_distance

                    paths[(area_names[i], area_names[j])] = {
                        'nodes': path_nodes,
                        'total_distance': total_distance,
                        'actual_distance': actual_distance,
                        'bike_lane_distance': bike_lane_distance,
                        'road_distance': actual_distance - bike_lane_distance,
                        'weighted_cost': path_weight
                    }

                    connected += 1

            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass

            if calculated % 5000 == 0:
                print(f"    Processed {calculated}/{total_pairs} pairs...")

    print(f"  Connected pairs: {connected}/{total_pairs} ({100*connected/total_pairs:.1f}%)")

    # Create DataFrames
    distance_df = pd.DataFrame(distance_matrix, index=area_names, columns=area_names)
    bike_lane_df = pd.DataFrame(bike_lane_matrix, index=area_names, columns=area_names)

    return distance_df, bike_lane_df, paths, area_names


def get_path_geometry(G, node_coords, path_nodes):
    """Get path geometry from node list."""
    if len(path_nodes) < 2:
        return None
    coords = [(node_coords[n][0], node_coords[n][1]) for n in path_nodes]
    return LineString(coords)


def export_results(distance_df, bike_lane_df, paths, G, node_coords, areas, output_dir, k_penalty):
    """Export results to files."""
    print(f"Exporting results to {output_dir}...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Distance matrix
    distance_df.to_csv(output_dir / "distance_matrix.csv")
    print("  Saved distance_matrix.csv")

    # 2. Bike lane usage matrix
    bike_lane_df.to_csv(output_dir / "bike_lane_distance_matrix.csv")
    print("  Saved bike_lane_distance_matrix.csv")

    # 3. Paths as GeoJSON
    path_features = []
    for (origin, dest), path_data in paths.items():
        geom = get_path_geometry(G, node_coords, path_data['nodes'])
        if geom:
            bike_pct = 100 * path_data['bike_lane_distance'] / path_data['actual_distance'] if path_data['actual_distance'] > 0 else 0
            path_features.append({
                'type': 'Feature',
                'properties': {
                    'origin': origin,
                    'destination': dest,
                    'total_distance_m': round(path_data['total_distance'], 2),
                    'bike_lane_distance_m': round(path_data['bike_lane_distance'], 2),
                    'road_distance_m': round(path_data['road_distance'], 2),
                    'bike_lane_pct': round(bike_pct, 1)
                },
                'geometry': geom.__geo_interface__
            })

    if path_features:
        paths_gdf = gpd.GeoDataFrame.from_features(path_features, crs=TARGET_CRS)
        paths_gdf = paths_gdf.to_crs(4326)
        paths_gdf.to_file(output_dir / "shortest_paths.geojson", driver='GeoJSON')
        print(f"  Saved shortest_paths.geojson ({len(path_features)} paths)")

    # 4. Area centers
    centers_gdf = areas.copy()
    centers_gdf['geometry'] = areas.geometry.centroid
    centers_gdf = centers_gdf.to_crs(4326)
    centers_gdf.to_file(output_dir / "area_centers.geojson", driver='GeoJSON')
    print("  Saved area_centers.geojson")

    # 5. Summary
    valid_distances = distance_df.values[(distance_df.values > 0) & (distance_df.values < np.inf)]
    valid_bike = bike_lane_df.values[(bike_lane_df.values >= 0) & (distance_df.values > 0) & (distance_df.values < np.inf)]

    avg_bike_pct = 100 * np.mean(valid_bike) / np.mean(valid_distances) if len(valid_distances) > 0 else 0

    summary = {
        'k_penalty': k_penalty,
        'total_areas': len(distance_df),
        'total_pairs': len(distance_df) * (len(distance_df) - 1) // 2,
        'connected_pairs': len(valid_distances) // 2,
        'connectivity_pct': round(100 * len(valid_distances) / (len(distance_df) * (len(distance_df) - 1)), 1),
        'avg_distance_m': round(float(np.mean(valid_distances)), 2) if len(valid_distances) > 0 else None,
        'min_distance_m': round(float(np.min(valid_distances)), 2) if len(valid_distances) > 0 else None,
        'max_distance_m': round(float(np.max(valid_distances)), 2) if len(valid_distances) > 0 else None,
        'median_distance_m': round(float(np.median(valid_distances)), 2) if len(valid_distances) > 0 else None,
        'avg_bike_lane_pct': round(avg_bike_pct, 1)
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print("  Saved summary.json")

    return summary


def main():
    print("=" * 70)
    print("BIKE PATH CALCULATOR WITH ROAD NETWORK")
    print(f"Road penalty factor K = {K_PENALTY}")
    print("=" * 70)

    # Load data
    bike_lanes = load_all_bike_lanes()
    areas = load_areas()

    # Transform to projected CRS
    print(f"\nTransforming to CRS {TARGET_CRS}...")
    bike_lanes = bike_lanes.to_crs(TARGET_CRS)
    areas = areas.to_crs(TARGET_CRS)

    # Create road network from area adjacencies
    print("\n" + "-" * 50)
    road_graph = create_road_network_from_areas(areas)

    # Create combined network
    print("\n" + "-" * 50)
    G, node_coords = create_combined_network(bike_lanes, road_graph, areas, K_PENALTY)

    # Calculate shortest paths
    print("\n" + "-" * 50)
    distance_df, bike_lane_df, paths, area_names = calculate_shortest_paths(
        G, node_coords, areas, K_PENALTY
    )

    # Export results
    print("\n" + "-" * 50)
    summary = export_results(
        distance_df, bike_lane_df, paths, G, node_coords, areas, OUTPUT_DIR, K_PENALTY
    )

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Road penalty factor K: {summary['k_penalty']}")
    print(f"Total areas: {summary['total_areas']}")
    print(f"Connected pairs: {summary['connected_pairs']}/{summary['total_pairs']} ({summary['connectivity_pct']}%)")

    if summary['avg_distance_m']:
        print(f"\nPath distances (meters):")
        print(f"  Average: {summary['avg_distance_m']:,.0f}")
        print(f"  Minimum: {summary['min_distance_m']:,.0f}")
        print(f"  Maximum: {summary['max_distance_m']:,.0f}")
        print(f"  Median:  {summary['median_distance_m']:,.0f}")
        print(f"\nAverage bike lane usage: {summary['avg_bike_lane_pct']:.1f}%")

    print(f"\nDistance Matrix (first 5x5):")
    print(distance_df.iloc[:5, :5].round(0).to_string())

    print(f"\nResults saved to: {OUTPUT_DIR}")

    return distance_df, paths


if __name__ == '__main__':
    distance_matrix, paths = main()
