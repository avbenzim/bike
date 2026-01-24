"""
Connect Bike Lane Network Gaps and Visualize

This script:
1. Loads bike lanes
2. Identifies gaps between lane endpoints
3. Connects nearby endpoints (within tolerance)
4. Visualizes the connected network
"""

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
from scipy.spatial import cKDTree
import fiona
import networkx as nx

# Configuration
script_dir = Path(__file__).parent
BIKE_LANES_COMPLETED = script_dir / "bike_lanes_completed.kml"
BIKE_LANES_CONSTRUCTION = script_dir / "bike_lanes_construction.kml"
AREAS_FILE = script_dir / "jer_areas.shp"
OUTPUT_FILE = script_dir / "jerusalem_bike_lanes_connected.png"

# Network connection parameters (in meters)
CONNECTION_TOLERANCE = 50  # Connect gaps smaller than this
ISOLATION_THRESHOLD = 200  # Lanes further than this are truly isolated

# Israel TM Grid CRS (meters)
TARGET_CRS = 2039

# Enable fiona KML driver
fiona.drvsupport.supported_drivers['KML'] = 'rw'


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
    """
    Connect nearby bike lane segments by adding short connecting lines.

    Only connects "dangling" endpoints (endpoints that don't already touch
    another lane) to their nearest neighbor within the tolerance.

    Args:
        bike_lanes: GeoDataFrame with bike lane geometries
        connection_tolerance: Maximum distance (meters) to bridge gaps

    Returns:
        List of connection dictionaries with geometry and metadata
    """
    print(f"Connecting network gaps (tolerance: {connection_tolerance}m)...")

    # Get all endpoints
    endpoints, endpoint_info = get_endpoints(bike_lanes)
    n_endpoints = len(endpoints)
    print(f"  Found {n_endpoints} endpoints from {len(bike_lanes)} lane segments")

    if n_endpoints == 0:
        return []

    # Create coordinate array for spatial indexing
    coords = np.array([[p.x, p.y] for p in endpoints])
    tree = cKDTree(coords)

    # First, identify which endpoints are already connected (touching another lane)
    # An endpoint is "connected" if there's another endpoint from a DIFFERENT lane
    # within a very small distance (1 meter)
    TOUCH_TOLERANCE = 1.0

    is_connected = [False] * n_endpoints
    for i in range(n_endpoints):
        nearby = tree.query_ball_point(coords[i], TOUCH_TOLERANCE)
        for j in nearby:
            if i != j and endpoint_info[i]['lane_idx'] != endpoint_info[j]['lane_idx']:
                is_connected[i] = True
                break

    n_dangling = sum(1 for c in is_connected if not c)
    print(f"  Found {n_dangling} dangling endpoints (not touching other lanes)")

    # For each dangling endpoint, find the nearest endpoint from a different lane
    connections = []
    already_connected_endpoints = set()

    for i in range(n_endpoints):
        if is_connected[i]:
            continue  # Skip already connected endpoints

        if i in already_connected_endpoints:
            continue  # Skip if we already created a connection for this endpoint

        # Find nearest neighbors within tolerance
        distances, indices = tree.query(coords[i], k=min(10, n_endpoints))

        best_j = None
        best_dist = float('inf')

        for dist, j in zip(distances, indices):
            if j == i:
                continue
            if endpoint_info[i]['lane_idx'] == endpoint_info[j]['lane_idx']:
                continue  # Same lane
            if dist > connection_tolerance:
                break  # Beyond tolerance
            if dist < 0.1:
                continue  # Already touching

            # Prefer connecting to other dangling endpoints
            if not is_connected[j] and j not in already_connected_endpoints:
                if dist < best_dist:
                    best_dist = dist
                    best_j = j

        # If no dangling neighbor found, try any endpoint
        if best_j is None:
            for dist, j in zip(distances, indices):
                if j == i:
                    continue
                if endpoint_info[i]['lane_idx'] == endpoint_info[j]['lane_idx']:
                    continue
                if dist > connection_tolerance:
                    break
                if dist < 0.1:
                    continue

                if dist < best_dist:
                    best_dist = dist
                    best_j = j
                    break  # Take the first valid one

        if best_j is not None:
            # Create connecting line
            connecting_line = LineString([
                (endpoints[i].x, endpoints[i].y),
                (endpoints[best_j].x, endpoints[best_j].y)
            ])

            connections.append({
                'geometry': connecting_line,
                'distance': best_dist,
                'from_lane': endpoint_info[i]['lane_idx'],
                'to_lane': endpoint_info[best_j]['lane_idx']
            })

            already_connected_endpoints.add(i)
            already_connected_endpoints.add(best_j)

    print(f"  Adding {len(connections)} connecting segments")

    return connections


def analyze_connectivity(bike_lanes):
    """Analyze network connectivity using graph components."""
    print("Analyzing network connectivity...")

    # Build a simple graph from endpoints
    endpoints, endpoint_info = get_endpoints(bike_lanes)

    if len(endpoints) == 0:
        return {'n_components': 0, 'largest_component': 0}

    # Create coordinate array
    coords = np.array([[p.x, p.y] for p in endpoints])
    tree = cKDTree(coords)

    # Build graph
    G = nx.Graph()

    # Add edges for each lane (connecting its endpoints)
    for i in range(0, len(endpoints), 2):
        if i + 1 < len(endpoints):
            G.add_edge(i, i + 1, lane=endpoint_info[i]['lane_idx'])

    # Connect endpoints that are very close (touching)
    for i in range(len(endpoints)):
        nearby = tree.query_ball_point(coords[i], 1.0)  # 1 meter tolerance
        for j in nearby:
            if i != j:
                G.add_edge(i, j)

    # Find connected components
    components = list(nx.connected_components(G))
    n_components = len(components)
    component_sizes = [len(c) for c in components]

    print(f"  Found {n_components} connected components")
    print(f"  Largest component: {max(component_sizes)} nodes")
    print(f"  Components with >10 nodes: {sum(1 for s in component_sizes if s > 10)}")

    return {
        'n_components': n_components,
        'largest_component': max(component_sizes),
        'component_sizes': component_sizes
    }


def main():
    print("=" * 60)
    print("BIKE LANE NETWORK CONNECTION ANALYSIS")
    print("=" * 60)

    print("\nLoading data...")

    # Load bike lanes
    bike_completed = gpd.read_file(BIKE_LANES_COMPLETED, driver='KML')
    bike_construction = gpd.read_file(BIKE_LANES_CONSTRUCTION, driver='KML')

    # Load areas
    areas = gpd.read_file(AREAS_FILE)
    areas = areas[areas['in_jeru'] == 1]

    print(f"Loaded {len(areas)} areas")
    print(f"Loaded {len(bike_completed)} completed bike lane segments")
    print(f"Loaded {len(bike_construction)} bike lanes under construction")

    # Combine all bike lanes
    all_lanes = gpd.GeoDataFrame(
        pd.concat([bike_completed, bike_construction], ignore_index=True),
        crs=bike_completed.crs
    )

    # Transform to projected CRS for accurate distance calculations
    print(f"\nTransforming to CRS {TARGET_CRS} for distance calculations...")
    all_lanes_proj = all_lanes.to_crs(TARGET_CRS)
    areas_proj = areas.to_crs(TARGET_CRS)

    # Analyze connectivity before
    print("\n--- Before connecting gaps ---")
    connectivity_before = analyze_connectivity(all_lanes_proj)

    # Connect gaps
    print("\n--- Connecting network gaps ---")
    connections = connect_network_gaps(all_lanes_proj, CONNECTION_TOLERANCE)

    # Create GeoDataFrame for connections
    if connections:
        connections_gdf = gpd.GeoDataFrame(
            connections,
            geometry=[c['geometry'] for c in connections],
            crs=TARGET_CRS
        )

        # Transform back to WGS84 for display
        connections_gdf = connections_gdf.to_crs(4326)
    else:
        connections_gdf = None

    # Transform areas back to WGS84
    areas = areas.to_crs(4326)
    bike_completed = bike_completed.to_crs(4326)
    bike_construction = bike_construction.to_crs(4326)

    # Create visualization
    print("\nCreating visualization...")

    fig, ax = plt.subplots(1, 1, figsize=(14, 12))

    # Plot areas as base layer
    areas.plot(ax=ax, facecolor='lightyellow', edgecolor='gray', linewidth=0.3, alpha=0.7)

    # Plot bike lanes under construction (orange dashed)
    bike_construction.plot(ax=ax, color='orange', linewidth=1.5, linestyle='--')

    # Plot completed bike lanes (green solid)
    bike_completed.plot(ax=ax, color='darkgreen', linewidth=1)

    # Plot connections (red)
    if connections_gdf is not None and len(connections_gdf) > 0:
        connections_gdf.plot(ax=ax, color='red', linewidth=2, linestyle='-')

    # Styling
    ax.set_title('Jerusalem Bike Lanes Network - Gap Connections', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)

    # Add legend
    legend_elements = [
        Line2D([0], [0], color='darkgreen', linewidth=2, label='Completed'),
        Line2D([0], [0], color='orange', linewidth=2, linestyle='--', label='Under Construction'),
        Line2D([0], [0], color='red', linewidth=2, label=f'Gap Connections (<{CONNECTION_TOLERANCE}m)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Grid
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()

    # Save
    plt.savefig(OUTPUT_FILE, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Map saved to: {OUTPUT_FILE}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Original segments: {len(bike_completed) + len(bike_construction)}")
    print(f"Gap connections added: {len(connections) if connections else 0}")
    print(f"Components before: {connectivity_before['n_components']}")

    plt.show()


if __name__ == '__main__':
    import pandas as pd
    main()
