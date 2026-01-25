"""
Create static map (PNG/PDF) of shortest path between Hebrew University and Navon Station
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

# Locations (WGS84)
HEBREW_UNIVERSITY_SCOPUS = (31.7939, 35.2433)  # lat, lon
NAVON_STATION = (31.7887, 35.2033)  # lat, lon

TARGET_CRS = 2039  # Israel TM Grid
K_PENALTY = 10

fiona.drvsupport.supported_drivers['KML'] = 'rw'


def load_data():
    """Load all data."""
    print("Loading data...")

    # Load areas
    areas = gpd.read_file(AREAS_FILE)
    areas = areas[areas['in_jeru'] == 1].copy()
    areas = areas.to_crs(TARGET_CRS)

    # Load roads
    print("  Loading roads...")
    df = pd.read_parquet(ROADS_FILE)
    df['geometry'] = df['geometry'].apply(lambda x: wkb.loads(x))
    roads = gpd.GeoDataFrame(df, geometry='geometry', crs=4326)
    roads = roads.to_crs(TARGET_CRS)

    jeru_bounds = areas.total_bounds
    buffer = 1000
    roads = roads.cx[jeru_bounds[0]-buffer:jeru_bounds[2]+buffer,
                     jeru_bounds[1]-buffer:jeru_bounds[3]+buffer]

    # Load bike lanes
    print("  Loading bike lanes...")
    completed = gpd.read_file(BIKE_LANES_COMPLETED, driver='KML').to_crs(TARGET_CRS)
    construction = gpd.read_file(BIKE_LANES_CONSTRUCTION, driver='KML').to_crs(TARGET_CRS)
    wishing = gpd.read_file(BIKE_LANES_WISHING_LIST, driver='KML').to_crs(TARGET_CRS)

    completed['source'] = 'completed'
    construction['source'] = 'construction'
    wishing['source'] = 'wishing'

    return roads, completed, construction, wishing, areas


def build_network(roads, bike_lanes_list, tolerance=10):
    """Build network graph."""
    print("Building network...")

    all_points = []

    for idx, row in roads.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or geom.geom_type != 'LineString':
            continue
        coords = list(geom.coords)
        if len(coords) >= 2:
            all_points.append((coords[0][0], coords[0][1]))
            all_points.append((coords[-1][0], coords[-1][1]))

    for bike_gdf in bike_lanes_list:
        for idx, row in bike_gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms)
            for line in lines:
                for coord in line.coords:
                    all_points.append((coord[0], coord[1]))

    coords_array = np.array(all_points)
    tree = cKDTree(coords_array)

    visited = set()
    node_coords = {}
    node_counter = 0

    for i in range(len(all_points)):
        if i in visited:
            continue
        neighbors = tree.query_ball_point(coords_array[i], tolerance)
        cluster_coords = coords_array[neighbors]
        centroid = cluster_coords.mean(axis=0)

        node_id = node_counter
        node_counter += 1
        node_coords[node_id] = (centroid[0], centroid[1])

        for j in neighbors:
            visited.add(j)

    G = nx.Graph()
    for nid, (x, y) in node_coords.items():
        G.add_node(nid, x=x, y=y)

    node_ids = list(node_coords.keys())
    node_coords_array = np.array([node_coords[n] for n in node_ids])
    node_tree = cKDTree(node_coords_array)

    # Add road edges
    for idx, row in roads.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or geom.geom_type != 'LineString':
            continue
        coords = list(geom.coords)
        if len(coords) < 2:
            continue

        _, start_idx = node_tree.query(coords[0][:2])
        _, end_idx = node_tree.query(coords[-1][:2])

        start_node = node_ids[start_idx]
        end_node = node_ids[end_idx]

        if start_node != end_node:
            length = geom.length
            if not G.has_edge(start_node, end_node) or G[start_node][end_node]['length'] > length:
                G.add_edge(start_node, end_node, length=length, has_bike_lane=False, source='road')

    # Add bike lane edges
    for bike_gdf in bike_lanes_list:
        source_name = bike_gdf['source'].iloc[0] if 'source' in bike_gdf.columns else 'bike'

        for idx, row in bike_gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms)

            for line in lines:
                coords = list(line.coords)
                if len(coords) < 2:
                    continue

                prev_node = None
                prev_coord = None
                for coord in coords:
                    x, y = coord[0], coord[1]
                    _, node_idx = node_tree.query([x, y])
                    node = node_ids[node_idx]

                    if prev_node is not None and prev_node != node:
                        segment = LineString([prev_coord[:2], (x, y)])
                        segment_length = segment.length

                        if G.has_edge(prev_node, node):
                            G[prev_node][node]['has_bike_lane'] = True
                            G[prev_node][node]['source'] = source_name
                        else:
                            G.add_edge(prev_node, node, length=segment_length, has_bike_lane=True, source=source_name)

                    prev_node = node
                    prev_coord = coord

    # Apply weights
    for u, v in G.edges():
        length = G[u][v]['length']
        has_bike = G[u][v].get('has_bike_lane', False)
        G[u][v]['weight'] = length if has_bike else length * K_PENALTY

    return G, node_coords


def connect_components(G, node_coords, max_gap=200):
    """Connect nearby disconnected components."""
    print("Connecting components...")

    components = list(nx.connected_components(G))
    if len(components) <= 1:
        return G

    component_boundaries = []
    for comp in components:
        comp_nodes = list(comp)
        comp_coords = np.array([node_coords[n] for n in comp_nodes])
        component_boundaries.append((comp_nodes, comp_coords))

    for i in range(len(component_boundaries)):
        nodes_i, coords_i = component_boundaries[i]
        tree_i = cKDTree(coords_i)

        for j in range(i + 1, len(component_boundaries)):
            nodes_j, coords_j = component_boundaries[j]

            min_dist = float('inf')
            best_pair = None

            for k, coord in enumerate(coords_j):
                dist, idx = tree_i.query(coord)
                if dist < min_dist:
                    min_dist = dist
                    best_pair = (nodes_i[idx], nodes_j[k])

            if min_dist < max_gap and best_pair:
                G.add_edge(best_pair[0], best_pair[1],
                          length=min_dist, weight=min_dist * K_PENALTY,
                          has_bike_lane=False, source='connection')

    return G


def find_shortest_path(G, node_coords, start_latlon, end_latlon):
    """Find shortest path."""
    start_point = gpd.GeoSeries([Point(start_latlon[1], start_latlon[0])], crs=4326).to_crs(TARGET_CRS)[0]
    end_point = gpd.GeoSeries([Point(end_latlon[1], end_latlon[0])], crs=4326).to_crs(TARGET_CRS)[0]

    node_ids = list(node_coords.keys())
    coords_array = np.array([node_coords[n] for n in node_ids])
    tree = cKDTree(coords_array)

    _, start_idx = tree.query([start_point.x, start_point.y])
    _, end_idx = tree.query([end_point.x, end_point.y])

    start_node = node_ids[start_idx]
    end_node = node_ids[end_idx]

    if nx.has_path(G, start_node, end_node):
        path = nx.shortest_path(G, start_node, end_node, weight='weight')
        actual_dist = sum(G[path[i]][path[i+1]]['length'] for i in range(len(path)-1))
        return path, actual_dist
    return None, None


def create_map(G, node_coords, path, completed, construction, wishing, areas):
    """Create static map."""
    print("Creating map...")

    fig, ax = plt.subplots(1, 1, figsize=(14, 12))

    # Convert locations to projected CRS
    start_pt = gpd.GeoSeries([Point(HEBREW_UNIVERSITY_SCOPUS[1], HEBREW_UNIVERSITY_SCOPUS[0])], crs=4326).to_crs(TARGET_CRS)[0]
    end_pt = gpd.GeoSeries([Point(NAVON_STATION[1], NAVON_STATION[0])], crs=4326).to_crs(TARGET_CRS)[0]

    # Set map bounds around the path
    all_x = [node_coords[n][0] for n in path] + [start_pt.x, end_pt.x]
    all_y = [node_coords[n][1] for n in path] + [start_pt.y, end_pt.y]
    buffer = 500
    ax.set_xlim(min(all_x) - buffer, max(all_x) + buffer)
    ax.set_ylim(min(all_y) - buffer, max(all_y) + buffer)

    # Plot areas (light gray background)
    areas.plot(ax=ax, color='#f0f0f0', edgecolor='#cccccc', linewidth=0.3)

    # Plot bike lanes
    completed.plot(ax=ax, color='green', linewidth=2, alpha=0.6, label='Completed')
    construction.plot(ax=ax, color='orange', linewidth=2, alpha=0.6, label='Construction')
    wishing.plot(ax=ax, color='purple', linewidth=2, alpha=0.6, label='Wishing List')

    # Plot path segments with colors based on type
    path_segments_bike = []
    path_segments_road = []
    path_segments_wishing = []

    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        x1, y1 = node_coords[u]
        x2, y2 = node_coords[v]
        segment = LineString([(x1, y1), (x2, y2)])

        has_bike = G[u][v].get('has_bike_lane', False)
        source = G[u][v].get('source', 'road')

        if has_bike:
            if source == 'wishing':
                path_segments_wishing.append(segment)
            else:
                path_segments_bike.append(segment)
        else:
            path_segments_road.append(segment)

    # Plot path segments
    if path_segments_bike:
        gdf_bike = gpd.GeoDataFrame(geometry=path_segments_bike, crs=TARGET_CRS)
        gdf_bike.plot(ax=ax, color='green', linewidth=5, alpha=0.9)

    if path_segments_wishing:
        gdf_wish = gpd.GeoDataFrame(geometry=path_segments_wishing, crs=TARGET_CRS)
        gdf_wish.plot(ax=ax, color='purple', linewidth=5, alpha=0.9)

    if path_segments_road:
        gdf_road = gpd.GeoDataFrame(geometry=path_segments_road, crs=TARGET_CRS)
        gdf_road.plot(ax=ax, color='red', linewidth=5, alpha=0.9)

    # Plot start and end markers
    ax.plot(start_pt.x, start_pt.y, 'o', markersize=15, color='blue', markeredgecolor='white', markeredgewidth=2, zorder=10)
    ax.plot(end_pt.x, end_pt.y, 's', markersize=15, color='red', markeredgecolor='white', markeredgewidth=2, zorder=10)

    # Add labels
    ax.annotate('Hebrew University\n(Mount Scopus)', xy=(start_pt.x, start_pt.y),
                xytext=(10, 10), textcoords='offset points', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.annotate('Navon Station', xy=(end_pt.x, end_pt.y),
                xytext=(10, -25), textcoords='offset points', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Calculate path statistics
    bike_dist = sum(G[path[i]][path[i+1]]['length'] for i in range(len(path)-1) if G[path[i]][path[i+1]].get('has_bike_lane', False))
    road_dist = sum(G[path[i]][path[i+1]]['length'] for i in range(len(path)-1) if not G[path[i]][path[i+1]].get('has_bike_lane', False))
    total_dist = bike_dist + road_dist

    # Legend
    legend_elements = [
        Line2D([0], [0], color='green', linewidth=3, label=f'Path on bike lane ({bike_dist/1000:.1f} km)'),
        Line2D([0], [0], color='red', linewidth=3, label=f'Path on road ({road_dist/1000:.1f} km)'),
        Line2D([0], [0], color='green', linewidth=2, alpha=0.6, label='Completed bike lanes'),
        Line2D([0], [0], color='orange', linewidth=2, alpha=0.6, label='Under construction'),
        Line2D([0], [0], color='purple', linewidth=2, alpha=0.6, label='Wishing list'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Start'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='End'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    # Title
    ax.set_title(f'Shortest Bike Path: Hebrew University → Navon Station\nTotal: {total_dist/1000:.1f} km | On bike lanes: {100*bike_dist/total_dist:.0f}%',
                 fontsize=14, fontweight='bold')

    ax.set_xlabel('X (meters)', fontsize=10)
    ax.set_ylabel('Y (meters)', fontsize=10)
    ax.set_aspect('equal')

    plt.tight_layout()
    return fig


def main():
    print("=" * 60)
    print("SHORTEST PATH MAP")
    print("=" * 60)

    roads, completed, construction, wishing, areas = load_data()
    G, node_coords = build_network(roads, [completed, construction, wishing])
    G = connect_components(G, node_coords)

    print("\nFinding path...")
    path, actual_dist = find_shortest_path(G, node_coords, HEBREW_UNIVERSITY_SCOPUS, NAVON_STATION)

    if path:
        print(f"Path found: {actual_dist/1000:.2f} km")

        fig = create_map(G, node_coords, path, completed, construction, wishing, areas)

        # Save as PNG and PDF
        png_path = script_dir / "shortest_path_map.png"
        pdf_path = script_dir / "shortest_path_map.pdf"

        fig.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
        fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')

        print(f"\nSaved: {png_path}")
        print(f"Saved: {pdf_path}")

        plt.close()
    else:
        print("No path found!")


if __name__ == '__main__':
    main()
