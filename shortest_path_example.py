"""
Visualize shortest path between Hebrew University (Mount Scopus) and Navon Station
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import folium
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

    # Load areas to get bounds
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
    print(f"  Roads: {len(roads)}")

    # Load bike lanes
    print("  Loading bike lanes...")
    completed = gpd.read_file(BIKE_LANES_COMPLETED, driver='KML').to_crs(TARGET_CRS)
    construction = gpd.read_file(BIKE_LANES_CONSTRUCTION, driver='KML').to_crs(TARGET_CRS)
    wishing = gpd.read_file(BIKE_LANES_WISHING_LIST, driver='KML').to_crs(TARGET_CRS)

    completed['source'] = 'completed'
    construction['source'] = 'construction'
    wishing['source'] = 'wishing'

    print(f"  Completed: {len(completed)}, Construction: {len(construction)}, Wishing: {len(wishing)}")

    return roads, completed, construction, wishing, areas


def build_network(roads, bike_lanes_list, tolerance=10):
    """Build network graph."""
    print("Building network...")

    # Collect all points
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

    print(f"  Total points: {len(all_points)}")

    # Cluster points
    coords_array = np.array(all_points)
    tree = cKDTree(coords_array)

    visited = set()
    node_coords = {}
    point_to_node = {}
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
            point_to_node[j] = node_id

    print(f"  Nodes: {len(node_coords)}")

    # Build graph
    G = nx.Graph()
    for nid, (x, y) in node_coords.items():
        G.add_node(nid, x=x, y=y)

    node_ids = list(node_coords.keys())
    node_coords_array = np.array([node_coords[n] for n in node_ids])
    node_tree = cKDTree(node_coords_array)

    # Store edge geometries for visualization
    edge_geoms = {}

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
                edge_geoms[(start_node, end_node)] = geom

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
                        edge_geoms[(prev_node, node)] = segment
                        edge_geoms[(node, prev_node)] = segment

                    prev_node = node
                    prev_coord = coord

    # Apply weights
    for u, v in G.edges():
        length = G[u][v]['length']
        has_bike = G[u][v].get('has_bike_lane', False)
        G[u][v]['weight'] = length if has_bike else length * K_PENALTY

    print(f"  Edges: {G.number_of_edges()}")

    return G, node_coords, edge_geoms


def connect_components(G, node_coords, max_gap=100):
    """Connect nearby disconnected components."""
    print("Connecting network components...")

    components = list(nx.connected_components(G))
    print(f"  Initial components: {len(components)}")

    if len(components) <= 1:
        return G

    # Get boundary nodes of each component
    component_boundaries = []
    for comp in components:
        comp_nodes = list(comp)
        comp_coords = np.array([node_coords[n] for n in comp_nodes])
        component_boundaries.append((comp_nodes, comp_coords))

    # Connect components
    connections = 0
    for i in range(len(component_boundaries)):
        nodes_i, coords_i = component_boundaries[i]
        tree_i = cKDTree(coords_i)

        for j in range(i + 1, len(component_boundaries)):
            nodes_j, coords_j = component_boundaries[j]

            # Find closest pair of nodes between components
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
                connections += 1

    print(f"  Added {connections} connections")
    print(f"  Final components: {nx.number_connected_components(G)}")

    return G


def find_shortest_path(G, node_coords, start_latlon, end_latlon):
    """Find shortest path between two lat/lon points."""

    # Convert lat/lon to projected coordinates
    start_point = gpd.GeoSeries([Point(start_latlon[1], start_latlon[0])], crs=4326).to_crs(TARGET_CRS)[0]
    end_point = gpd.GeoSeries([Point(end_latlon[1], end_latlon[0])], crs=4326).to_crs(TARGET_CRS)[0]

    # Find nearest nodes
    node_ids = list(node_coords.keys())
    coords_array = np.array([node_coords[n] for n in node_ids])
    tree = cKDTree(coords_array)

    _, start_idx = tree.query([start_point.x, start_point.y])
    _, end_idx = tree.query([end_point.x, end_point.y])

    start_node = node_ids[start_idx]
    end_node = node_ids[end_idx]

    print(f"Start node: {start_node} at {node_coords[start_node]}")
    print(f"End node: {end_node} at {node_coords[end_node]}")

    # Check connectivity
    start_comp = nx.node_connected_component(G, start_node)
    end_comp = nx.node_connected_component(G, end_node)
    print(f"Start component size: {len(start_comp)}")
    print(f"End component size: {len(end_comp)}")
    print(f"Same component: {start_node in end_comp}")

    # Find path
    if nx.has_path(G, start_node, end_node):
        path = nx.shortest_path(G, start_node, end_node, weight='weight')
        distance = nx.shortest_path_length(G, start_node, end_node, weight='weight')
        print(f"Path found: {len(path)} nodes, {distance/1000:.2f} km (weighted)")

        # Calculate actual distance
        actual_dist = sum(G[path[i]][path[i+1]]['length'] for i in range(len(path)-1))
        print(f"Actual distance: {actual_dist/1000:.2f} km")

        return path, distance, actual_dist
    else:
        print("No path found!")
        return None, None, None


def create_map(G, node_coords, edge_geoms, path, completed, construction, wishing):
    """Create a folium map with the path and bike lanes."""

    # Center of map
    center = [(HEBREW_UNIVERSITY_SCOPUS[0] + NAVON_STATION[0]) / 2,
              (HEBREW_UNIVERSITY_SCOPUS[1] + NAVON_STATION[1]) / 2]

    m = folium.Map(location=center, zoom_start=14, tiles='cartodbpositron')

    # Add bike lanes
    def add_bike_lanes(gdf, color, name):
        gdf_wgs = gdf.to_crs(4326)
        for idx, row in gdf_wgs.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms)
            for line in lines:
                coords = [(c[1], c[0]) for c in line.coords]
                folium.PolyLine(coords, weight=3, color=color, opacity=0.7,
                               popup=name).add_to(m)

    add_bike_lanes(completed, 'green', 'Completed')
    add_bike_lanes(construction, 'orange', 'Construction')
    add_bike_lanes(wishing, 'purple', 'Wishing List')

    # Add path
    if path:
        path_coords = []
        for i, node in enumerate(path):
            x, y = node_coords[node]
            # Convert to WGS84
            point = gpd.GeoSeries([Point(x, y)], crs=TARGET_CRS).to_crs(4326)[0]
            path_coords.append((point.y, point.x))

        # Draw path segments with different colors based on bike lane status
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            has_bike = G[u][v].get('has_bike_lane', False)
            source = G[u][v].get('source', 'road')

            if has_bike:
                if source == 'wishing':
                    color = 'purple'
                elif source == 'construction':
                    color = 'orange'
                else:
                    color = 'green'
            else:
                color = 'red'

            x1, y1 = node_coords[u]
            x2, y2 = node_coords[v]
            p1 = gpd.GeoSeries([Point(x1, y1)], crs=TARGET_CRS).to_crs(4326)[0]
            p2 = gpd.GeoSeries([Point(x2, y2)], crs=TARGET_CRS).to_crs(4326)[0]

            folium.PolyLine([(p1.y, p1.x), (p2.y, p2.x)],
                           weight=6, color=color, opacity=0.9).add_to(m)

    # Add markers
    folium.Marker(
        HEBREW_UNIVERSITY_SCOPUS,
        popup='Hebrew University (Mount Scopus)',
        icon=folium.Icon(color='blue', icon='graduation-cap', prefix='fa')
    ).add_to(m)

    folium.Marker(
        NAVON_STATION,
        popup='Navon Central Station',
        icon=folium.Icon(color='red', icon='train', prefix='fa')
    ).add_to(m)

    # Add legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                background-color: white; padding: 10px; border: 2px solid grey;
                border-radius: 5px; font-size: 14px;">
        <b>Legend</b><br>
        <span style="color: green;">━━</span> Completed bike lane<br>
        <span style="color: orange;">━━</span> Under construction<br>
        <span style="color: purple;">━━</span> Wishing list<br>
        <span style="color: red;">━━</span> Road (no bike lane)<br>
        <span style="color: blue;">●</span> Start<br>
        <span style="color: red;">●</span> End
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


def main():
    print("=" * 60)
    print("SHORTEST PATH: Hebrew University → Navon Station")
    print("=" * 60)

    # Load data
    roads, completed, construction, wishing, areas = load_data()

    # Build network with ALL bike lanes (including wishing list)
    G, node_coords, edge_geoms = build_network(roads, [completed, construction, wishing])

    # Connect disconnected components
    G = connect_components(G, node_coords, max_gap=200)

    # Find shortest path
    print("\nFinding shortest path...")
    path, weighted_dist, actual_dist = find_shortest_path(
        G, node_coords,
        HEBREW_UNIVERSITY_SCOPUS,
        NAVON_STATION
    )

    if path:
        # Analyze path
        bike_lane_dist = 0
        road_dist = 0

        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            length = G[u][v]['length']
            if G[u][v].get('has_bike_lane', False):
                bike_lane_dist += length
            else:
                road_dist += length

        print(f"\nPath Analysis:")
        print(f"  Total distance: {actual_dist/1000:.2f} km")
        print(f"  On bike lanes: {bike_lane_dist/1000:.2f} km ({100*bike_lane_dist/actual_dist:.1f}%)")
        print(f"  On roads: {road_dist/1000:.2f} km ({100*road_dist/actual_dist:.1f}%)")

        # Create map
        print("\nCreating map...")
        m = create_map(G, node_coords, edge_geoms, path, completed, construction, wishing)

        output_file = script_dir / "shortest_path_map.html"
        m.save(str(output_file))
        print(f"Map saved to: {output_file}")

    return path


if __name__ == '__main__':
    path = main()
