"""
Bike Path Shortest Route Calculator for Jerusalem

This script calculates the shortest bike path between the centers of areas
using bike lane network data.

Input:
    - Bike lanes shapefile/GeoJSON (line geometries representing bike paths)
    - Areas shapefile (polygon geometries representing areas/neighborhoods)

Output:
    - Shortest paths between all pairs of area centers
    - Distance matrix
    - Optional: Path geometries as GeoJSON
"""

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import nearest_points, unary_union
from scipy.spatial import cKDTree
import json
import argparse
from pathlib import Path
from itertools import combinations
import warnings

warnings.filterwarnings('ignore')


def load_bike_lanes(file_path: str) -> gpd.GeoDataFrame:
    """
    Load bike lanes from shapefile or GeoJSON.

    Args:
        file_path: Path to bike lanes file (.shp or .geojson)

    Returns:
        GeoDataFrame with bike lane geometries
    """
    file_path = Path(file_path)

    if file_path.suffix.lower() in ['.shp', '.geojson', '.json']:
        gdf = gpd.read_file(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    # Ensure we have line geometries
    valid_geom_types = ['LineString', 'MultiLineString']
    gdf = gdf[gdf.geometry.type.isin(valid_geom_types)]

    if gdf.empty:
        raise ValueError("No valid line geometries found in bike lanes file")

    print(f"Loaded {len(gdf)} bike lane segments")
    return gdf


def load_areas(file_path: str) -> gpd.GeoDataFrame:
    """
    Load areas from shapefile.

    Args:
        file_path: Path to areas shapefile

    Returns:
        GeoDataFrame with area polygons
    """
    gdf = gpd.read_file(file_path)

    # Ensure we have polygon geometries
    valid_geom_types = ['Polygon', 'MultiPolygon']
    gdf = gdf[gdf.geometry.type.isin(valid_geom_types)]

    if gdf.empty:
        raise ValueError("No valid polygon geometries found in areas file")

    print(f"Loaded {len(gdf)} areas")
    return gdf


def get_area_centers(areas_gdf: gpd.GeoDataFrame, name_column: str = None) -> gpd.GeoDataFrame:
    """
    Calculate the centroid of each area.

    Args:
        areas_gdf: GeoDataFrame with area polygons
        name_column: Optional column name for area names

    Returns:
        GeoDataFrame with area centroids
    """
    centers = areas_gdf.copy()
    centers['centroid'] = centers.geometry.centroid
    centers = centers.set_geometry('centroid')

    # Create area IDs if no name column specified
    if name_column and name_column in centers.columns:
        centers['area_name'] = centers[name_column]
    else:
        centers['area_name'] = [f"Area_{i}" for i in range(len(centers))]

    print(f"Calculated {len(centers)} area centers")
    return centers


def build_network_graph(bike_lanes: gpd.GeoDataFrame, tolerance: float = 1.0) -> nx.Graph:
    """
    Build a NetworkX graph from bike lane geometries.

    Args:
        bike_lanes: GeoDataFrame with bike lane line geometries
        tolerance: Distance tolerance for snapping nodes (in CRS units)

    Returns:
        NetworkX Graph representing the bike network
    """
    G = nx.Graph()

    # Extract all coordinates from line geometries
    all_coords = []
    edge_data = []

    for idx, row in bike_lanes.iterrows():
        geom = row.geometry

        if geom.geom_type == 'MultiLineString':
            lines = list(geom.geoms)
        else:
            lines = [geom]

        for line in lines:
            coords = list(line.coords)
            if len(coords) >= 2:
                start = coords[0]
                end = coords[-1]
                length = line.length

                all_coords.append(start)
                all_coords.append(end)
                edge_data.append({
                    'start': start,
                    'end': end,
                    'length': length,
                    'geometry': line
                })

    # Snap nodes that are within tolerance
    coords_array = np.array(all_coords)
    if len(coords_array) > 0:
        tree = cKDTree(coords_array)

        # Create node mapping for snapped coordinates
        node_map = {}
        node_counter = 0

        for i, coord in enumerate(all_coords):
            coord_tuple = tuple(coord)
            if coord_tuple not in node_map:
                # Check for nearby existing nodes
                nearby = tree.query_ball_point(coord, tolerance)
                found_existing = False

                for j in nearby:
                    if tuple(all_coords[j]) in node_map:
                        node_map[coord_tuple] = node_map[tuple(all_coords[j])]
                        found_existing = True
                        break

                if not found_existing:
                    node_map[coord_tuple] = node_counter
                    G.add_node(node_counter, x=coord[0], y=coord[1])
                    node_counter += 1

    # Add edges
    for edge in edge_data:
        start_node = node_map.get(tuple(edge['start']))
        end_node = node_map.get(tuple(edge['end']))

        if start_node is not None and end_node is not None and start_node != end_node:
            # Add edge with length as weight
            if G.has_edge(start_node, end_node):
                # Keep the shorter edge
                if G[start_node][end_node]['weight'] > edge['length']:
                    G[start_node][end_node]['weight'] = edge['length']
                    G[start_node][end_node]['geometry'] = edge['geometry']
            else:
                G.add_edge(start_node, end_node,
                          weight=edge['length'],
                          geometry=edge['geometry'])

    print(f"Built network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G


def find_nearest_network_node(point: Point, G: nx.Graph) -> int:
    """
    Find the nearest network node to a given point.

    Args:
        point: Shapely Point geometry
        G: NetworkX Graph with node coordinates

    Returns:
        Node ID of the nearest network node
    """
    nodes_coords = np.array([[G.nodes[n]['x'], G.nodes[n]['y']] for n in G.nodes()])
    nodes_list = list(G.nodes())

    if len(nodes_coords) == 0:
        return None

    tree = cKDTree(nodes_coords)
    dist, idx = tree.query([point.x, point.y])

    return nodes_list[idx]


def calculate_shortest_paths(
    G: nx.Graph,
    centers: gpd.GeoDataFrame
) -> tuple[pd.DataFrame, dict]:
    """
    Calculate shortest paths between all pairs of area centers.

    Args:
        G: NetworkX Graph of bike network
        centers: GeoDataFrame with area centroids

    Returns:
        Tuple of (distance_matrix DataFrame, paths dictionary)
    """
    # Find nearest network node for each center
    center_nodes = {}
    for idx, row in centers.iterrows():
        centroid = row.geometry
        nearest_node = find_nearest_network_node(centroid, G)
        center_nodes[row['area_name']] = nearest_node

    area_names = list(center_nodes.keys())
    n_areas = len(area_names)

    # Initialize distance matrix
    distance_matrix = np.full((n_areas, n_areas), np.inf)
    np.fill_diagonal(distance_matrix, 0)

    # Store paths
    paths = {}

    # Calculate shortest paths between all pairs
    for i, area1 in enumerate(area_names):
        for j, area2 in enumerate(area_names):
            if i >= j:
                continue

            node1 = center_nodes[area1]
            node2 = center_nodes[area2]

            if node1 is None or node2 is None:
                print(f"Warning: Could not find network node for {area1 if node1 is None else area2}")
                continue

            try:
                path = nx.shortest_path(G, node1, node2, weight='weight')
                path_length = nx.shortest_path_length(G, node1, node2, weight='weight')

                distance_matrix[i, j] = path_length
                distance_matrix[j, i] = path_length

                paths[(area1, area2)] = {
                    'nodes': path,
                    'length': path_length
                }
                paths[(area2, area1)] = {
                    'nodes': list(reversed(path)),
                    'length': path_length
                }

            except nx.NetworkXNoPath:
                print(f"No path found between {area1} and {area2}")

    # Create DataFrame
    distance_df = pd.DataFrame(
        distance_matrix,
        index=area_names,
        columns=area_names
    )

    return distance_df, paths


def get_path_geometry(G: nx.Graph, path_nodes: list) -> LineString:
    """
    Reconstruct path geometry from node sequence.

    Args:
        G: NetworkX Graph
        path_nodes: List of node IDs in path order

    Returns:
        LineString geometry of the path
    """
    coords = []

    for i, node in enumerate(path_nodes):
        coords.append((G.nodes[node]['x'], G.nodes[node]['y']))

    if len(coords) >= 2:
        return LineString(coords)
    return None


def export_results(
    distance_matrix: pd.DataFrame,
    paths: dict,
    G: nx.Graph,
    output_dir: str,
    centers: gpd.GeoDataFrame
):
    """
    Export results to files.

    Args:
        distance_matrix: DataFrame with distances between areas
        paths: Dictionary of shortest paths
        G: NetworkX Graph
        output_dir: Output directory path
        centers: GeoDataFrame with area centroids
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Export distance matrix
    distance_matrix.to_csv(output_path / 'distance_matrix.csv')
    print(f"Saved distance matrix to {output_path / 'distance_matrix.csv'}")

    # Export paths as GeoJSON
    path_features = []
    for (origin, dest), path_data in paths.items():
        if origin < dest:  # Avoid duplicates
            geom = get_path_geometry(G, path_data['nodes'])
            if geom:
                path_features.append({
                    'type': 'Feature',
                    'properties': {
                        'origin': origin,
                        'destination': dest,
                        'length_meters': path_data['length']
                    },
                    'geometry': geom.__geo_interface__
                })

    paths_geojson = {
        'type': 'FeatureCollection',
        'features': path_features
    }

    with open(output_path / 'shortest_paths.geojson', 'w') as f:
        json.dump(paths_geojson, f, indent=2)
    print(f"Saved path geometries to {output_path / 'shortest_paths.geojson'}")

    # Export area centers
    centers_export = centers[['area_name', 'geometry']].copy()
    centers_export = centers_export.set_geometry('geometry')
    centers_export.to_file(output_path / 'area_centers.geojson', driver='GeoJSON')
    print(f"Saved area centers to {output_path / 'area_centers.geojson'}")

    # Export summary statistics
    summary = {
        'total_areas': len(distance_matrix),
        'total_path_pairs': len(paths) // 2,
        'average_path_length': np.nanmean(distance_matrix.values[distance_matrix.values < np.inf]),
        'max_path_length': np.nanmax(distance_matrix.values[distance_matrix.values < np.inf]),
        'min_path_length': np.nanmin(distance_matrix.values[(distance_matrix.values > 0) & (distance_matrix.values < np.inf)])
    }

    with open(output_path / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {output_path / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate shortest bike paths between area centers in Jerusalem'
    )
    parser.add_argument(
        'bike_lanes',
        help='Path to bike lanes file (shapefile or GeoJSON)'
    )
    parser.add_argument(
        'areas',
        help='Path to areas shapefile'
    )
    parser.add_argument(
        '-o', '--output',
        default='./output',
        help='Output directory (default: ./output)'
    )
    parser.add_argument(
        '-n', '--name-column',
        help='Column name for area names in the areas shapefile'
    )
    parser.add_argument(
        '-t', '--tolerance',
        type=float,
        default=1.0,
        help='Node snapping tolerance in CRS units (default: 1.0)'
    )
    parser.add_argument(
        '--crs',
        help='Target CRS for distance calculations (e.g., EPSG:2039 for Israel)'
    )

    args = parser.parse_args()

    # Load data
    print("Loading bike lanes...")
    bike_lanes = load_bike_lanes(args.bike_lanes)

    print("Loading areas...")
    areas = load_areas(args.areas)

    # Reproject if needed
    if args.crs:
        target_crs = args.crs
    else:
        # Default to Israel TM Grid (EPSG:2039) for accurate distance calculations
        target_crs = 'EPSG:2039'

    print(f"Reprojecting to {target_crs}...")
    bike_lanes = bike_lanes.to_crs(target_crs)
    areas = areas.to_crs(target_crs)

    # Calculate area centers
    print("Calculating area centers...")
    centers = get_area_centers(areas, args.name_column)

    # Build network graph
    print("Building bike network graph...")
    G = build_network_graph(bike_lanes, tolerance=args.tolerance)

    if G.number_of_nodes() == 0:
        print("Error: No network nodes created. Check your bike lanes data.")
        return

    # Calculate shortest paths
    print("Calculating shortest paths...")
    distance_matrix, paths = calculate_shortest_paths(G, centers)

    # Export results
    print("Exporting results...")
    export_results(distance_matrix, paths, G, args.output, centers)

    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Areas processed: {len(centers)}")
    print(f"Path pairs calculated: {len(paths) // 2}")

    valid_distances = distance_matrix.values[(distance_matrix.values > 0) & (distance_matrix.values < np.inf)]
    if len(valid_distances) > 0:
        print(f"Average path length: {np.mean(valid_distances):.2f} meters")
        print(f"Shortest path: {np.min(valid_distances):.2f} meters")
        print(f"Longest path: {np.max(valid_distances):.2f} meters")

    print("\nDistance Matrix (first 5x5):")
    print(distance_matrix.iloc[:5, :5].to_string())


if __name__ == '__main__':
    main()
