"""
Show example trips from Ramot that use the Bar Lev lane.
"""

import geopandas as gpd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
import fiona
import warnings

warnings.filterwarnings('ignore')

script_dir = Path(__file__).parent
ROADS_FILE = script_dir / "jerusalem_roads.kml"
BIKE_LANES_COMPLETED = script_dir / "bike_lanes_completed.kml"
BIKE_LANES_CONSTRUCTION = script_dir / "bike_lanes_construction.kml"
BIKE_LANES_WISHING_LIST = script_dir / "bike_lanes_wishing_list.kml"
AREAS_FILE = script_dir / "jer_areas.shp"

TARGET_CRS = 2039
NODE_TOLERANCE = 15
SNAP_TOLERANCE = 50
K_PENALTY = 100

fiona.drvsupport.supported_drivers['KML'] = 'rw'


def build_network(roads_gdf, bike_lanes_list, tolerance=15, snap_tolerance=50):
    """Build NetworkX graph with proper snapping."""
    G = nx.Graph()
    coord_to_node = {}
    node_coords = {}
    node_counter = [0]

    def get_or_create_node_grid(x, y):
        key = (round(x / tolerance) * tolerance, round(y / tolerance) * tolerance)
        if key in coord_to_node:
            return coord_to_node[key]
        nid = node_counter[0]
        node_counter[0] += 1
        coord_to_node[key] = nid
        node_coords[nid] = (x, y)
        G.add_node(nid, x=x, y=y)
        return nid

    # Add roads
    for idx, row in roads_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or geom.geom_type != 'LineString':
            continue
        coords = list(geom.coords)
        if len(coords) >= 2:
            start = get_or_create_node_grid(coords[0][0], coords[0][1])
            end = get_or_create_node_grid(coords[-1][0], coords[-1][1])
            if start != end:
                G.add_edge(start, end, length=geom.length, has_bike_lane=False)

    # Build spatial index
    node_ids = list(node_coords.keys())
    coords_array = np.array([node_coords[n] for n in node_ids])
    node_tree = cKDTree(coords_array)

    def get_nearest_node(x, y):
        if len(coords_array) > 0:
            dist, idx = node_tree.query([x, y])
            if dist <= snap_tolerance:
                return node_ids[idx]
        return get_or_create_node_grid(x, y)

    # Add bike lanes
    for bl_gdf in bike_lanes_list:
        if bl_gdf is None:
            continue
        bl_proj = bl_gdf.to_crs(TARGET_CRS)
        for idx, row in bl_proj.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty or geom.geom_type != 'LineString':
                continue
            coords = list(geom.coords)
            if len(coords) >= 2:
                start = get_nearest_node(coords[0][0], coords[0][1])
                end = get_nearest_node(coords[-1][0], coords[-1][1])
                if start != end:
                    if G.has_edge(start, end):
                        G[start][end]['has_bike_lane'] = True
                    else:
                        G.add_edge(start, end, length=geom.length, has_bike_lane=True)

    # Rebuild index
    node_ids = list(node_coords.keys())
    coords_array = np.array([node_coords[n] for n in node_ids])
    node_tree = cKDTree(coords_array)

    return G, node_coords, node_tree, node_ids


def apply_weights(G, k_penalty):
    G = G.copy()
    for u, v in G.edges():
        length = G[u][v]['length']
        has_bike = G[u][v].get('has_bike_lane', False)
        G[u][v]['weight'] = length if has_bike else length * k_penalty
    return G


def get_path_geometry(G, node_coords, path):
    """Convert path to LineString."""
    coords = [(node_coords[n][0], node_coords[n][1]) for n in path]
    return LineString(coords)


def check_path_uses_bar_lev(G, node_coords, path, bar_lev_nodes):
    """Check if path uses Bar Lev lane nodes."""
    path_set = set(path)
    return len(path_set.intersection(bar_lev_nodes)) >= 2


def main():
    print("=" * 70)
    print("RAMOT TRIPS THROUGH BAR LEV LANE")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    areas = gpd.read_file(AREAS_FILE)
    areas = areas[areas['in_jeru'] == 1].copy()
    areas = areas.to_crs(TARGET_CRS)

    roads = gpd.read_file(ROADS_FILE, driver='KML').to_crs(TARGET_CRS)
    completed = gpd.read_file(BIKE_LANES_COMPLETED, driver='KML')
    construction = gpd.read_file(BIKE_LANES_CONSTRUCTION, driver='KML')
    wishing = gpd.read_file(BIKE_LANES_WISHING_LIST, driver='KML')

    bar_lev = wishing[wishing['Name'].str.contains('בר לב', na=False)]

    # Find Ramot areas
    ramot_areas = areas[areas['name'].str.contains('רמות', na=False)]
    print(f"  Found {len(ramot_areas)} Ramot areas")

    # Pick central Ramot area (with highest population)
    ramot_areas_sorted = ramot_areas.sort_values('pop_2025', ascending=False)
    ramot_origin = ramot_areas_sorted.iloc[0]
    ramot_centroid = ramot_origin.geometry.centroid
    print(f"  Using: {ramot_origin['name']} (pop: {ramot_origin['pop_2025']:.0f})")

    # Build network WITH Bar Lev
    print("\nBuilding network with Bar Lev...")
    G, node_coords, node_tree, node_ids = build_network(
        roads, [completed, construction, bar_lev], NODE_TOLERANCE, SNAP_TOLERANCE
    )
    G_weighted = apply_weights(G, K_PENALTY)

    # Find Bar Lev lane nodes
    bar_lev_proj = bar_lev.to_crs(TARGET_CRS)
    bar_lev_coords = list(bar_lev_proj.geometry.iloc[0].coords)
    bar_lev_nodes = set()
    for coord in bar_lev_coords:
        _, idx = node_tree.query([coord[0], coord[1]])
        bar_lev_nodes.add(node_ids[idx])
    print(f"  Bar Lev has {len(bar_lev_nodes)} nodes")

    # Find Ramot node
    _, ramot_idx = node_tree.query([ramot_centroid.x, ramot_centroid.y])
    ramot_node = node_ids[ramot_idx]

    # Find destinations - pick diverse areas across the city
    destinations = [
        ('מרכז העיר', areas[areas['name'].str.contains('מרכז העיר|בן יהודה', na=False)]),
        ('גבעת רם', areas[areas['name'].str.contains('גבעת רם', na=False)]),
        ('תלפיות', areas[areas['name'].str.contains('תלפיות', na=False)]),
        ('קטמון', areas[areas['name'].str.contains('קטמון', na=False)]),
        ('רחביה', areas[areas['name'].str.contains('רחביה', na=False)]),
        ('בית הכרם', areas[areas['name'].str.contains('בית הכרם', na=False)]),
        ('נווה יעקב', areas[areas['name'].str.contains('נווה יעקב', na=False)]),
        ('גילה', areas[areas['name'].str.contains('גילה', na=False)]),
    ]

    # Calculate paths and find ones that use Bar Lev
    print("\nFinding paths from Ramot through Bar Lev...")
    trips_through_bar_lev = []

    for dest_name, dest_areas in destinations:
        if len(dest_areas) == 0:
            continue
        dest_area = dest_areas.sort_values('emp_2025', ascending=False).iloc[0]
        dest_centroid = dest_area.geometry.centroid
        _, dest_idx = node_tree.query([dest_centroid.x, dest_centroid.y])
        dest_node = node_ids[dest_idx]

        try:
            path = nx.shortest_path(G_weighted, ramot_node, dest_node, weight='weight')
            dist = nx.shortest_path_length(G_weighted, ramot_node, dest_node, weight='weight')

            uses_bar_lev = check_path_uses_bar_lev(G, node_coords, path, bar_lev_nodes)

            if uses_bar_lev:
                path_geom = get_path_geometry(G, node_coords, path)
                trips_through_bar_lev.append({
                    'dest_name': dest_name,
                    'dest_area_name': dest_area['name'],
                    'path': path,
                    'geometry': path_geom,
                    'distance': dist / 1000,  # km
                    'dest_centroid': dest_centroid
                })
                print(f"  ✓ {dest_name} ({dest_area['name'][:30]}): {dist/1000:.1f}km - USES BAR LEV")
            else:
                print(f"  ✗ {dest_name} ({dest_area['name'][:30]}): {dist/1000:.1f}km - does not use Bar Lev")

        except nx.NetworkXNoPath:
            print(f"  - {dest_name}: No path found")

    print(f"\n{len(trips_through_bar_lev)} trips use Bar Lev lane")

    # Create map
    if trips_through_bar_lev:
        print("\nCreating map...")
        fig, ax = plt.subplots(figsize=(14, 16))

        # Plot areas
        areas.plot(ax=ax, color='#F0F0F0', edgecolor='#CCCCCC', linewidth=0.3)

        # Highlight Ramot areas
        ramot_areas.plot(ax=ax, color='#90EE90', edgecolor='#228B22', linewidth=1)

        # Plot existing bike lanes in gray
        for bl in [completed, construction]:
            bl.to_crs(TARGET_CRS).plot(ax=ax, color='gray', linewidth=1, alpha=0.5)

        # Plot Bar Lev in red (thick)
        bar_lev_proj.plot(ax=ax, color='red', linewidth=4, alpha=0.9, label='Bar Lev Lane')

        # Plot trips with different colors
        colors = ['#0000FF', '#FF00FF', '#00FFFF', '#FFA500', '#800080']
        for i, trip in enumerate(trips_through_bar_lev):
            color = colors[i % len(colors)]
            gpd.GeoDataFrame({'geometry': [trip['geometry']]}, crs=TARGET_CRS).plot(
                ax=ax, color=color, linewidth=2.5, alpha=0.8,
                label=f"{trip['dest_name']} ({trip['distance']:.1f}km)"
            )
            # Mark destination
            ax.plot(trip['dest_centroid'].x, trip['dest_centroid'].y,
                   'o', color=color, markersize=10, markeredgecolor='black')

        # Mark Ramot origin
        ax.plot(ramot_centroid.x, ramot_centroid.y, '*', color='green',
               markersize=20, markeredgecolor='black', label='Ramot (origin)')

        ax.set_title(f'Trips from Ramot ({ramot_origin["name"]}) through Bar Lev Lane',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.set_xlabel('X (Israel TM Grid)')
        ax.set_ylabel('Y (Israel TM Grid)')

        plt.tight_layout()
        plt.savefig(script_dir / 'ramot_bar_lev_trips.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: ramot_bar_lev_trips.png")

    print("\nDone!")


if __name__ == '__main__':
    main()
