"""Generate a single self-contained interactive HTML map for bike lane analysis.
All data is embedded in the HTML - no external file loading needed.
"""
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import json
import subprocess
from pathlib import Path
from scipy.spatial import cKDTree
from pyproj import Transformer
import fiona
import warnings

warnings.filterwarnings('ignore')

def get_version():
    """Get version from git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, cwd=script_dir
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    return 'dev'

script_dir = Path(__file__).parent
fiona.drvsupport.supported_drivers['KML'] = 'rw'
TARGET_CRS = 2039
WGS84 = 4326
NODE_TOLERANCE = 15

# Default values for dropdowns (more options, user can also enter custom)
K_VALUES = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
THETA_VALUES = [-0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -2.0, -2.5, -3.0]
DATA_YEARS = [2020, 2025, 2030, 2035, 2040]
DEFAULT_YEAR = 2025

# Road discomfort factors by OSM highway type
DISCOMFORT_FACTORS = {
    'residential': 1.0, 'living_street': 1.0,
    'tertiary': 1.5, 'tertiary_link': 1.5,
    'secondary': 2.0, 'secondary_link': 2.0,
    'primary': 3.0, 'primary_link': 3.0,
    'trunk': 4.0, 'trunk_link': 4.0,
    'motorway': 5.0, 'motorway_link': 5.0,
}

# Academic institutions with student counts and coordinates (WGS84)
INSTITUTIONS = [
    {"name": "Hebrew University Mt Scopus", "lat": 31.7927, "lon": 35.2454, "students": 11500},
    {"name": "Hebrew University Givat Ram", "lat": 31.7730, "lon": 35.1972, "students": 8000},
    {"name": "Hebrew University Ein Kerem", "lat": 31.7640, "lon": 35.1500, "students": 3500},
    {"name": "Bezalel New Campus", "lat": 31.7827, "lon": 35.2266, "students": 3500},
    {"name": "Hadassah College", "lat": 31.7831, "lon": 35.2211, "students": 4600},
    {"name": "Machon Lev", "lat": 31.7654, "lon": 35.1910, "students": 3500},
    {"name": "David Yellin", "lat": 31.7821, "lon": 35.1906, "students": 3000},
    {"name": "Azrieli Engineering", "lat": 31.7685, "lon": 35.1939, "students": 2800},
    {"name": "Music Academy", "lat": 31.7773, "lon": 35.1948, "students": 700},
    {"name": "Sam Spiegel", "lat": 31.7798, "lon": 35.2130, "students": 300},
    {"name": "Musrara", "lat": 31.7827, "lon": 35.2266, "students": 200},
    {"name": "Al-Quds University", "lat": 31.8313, "lon": 35.2250, "students": 12000},
]

TRAIN_STATIONS = [
    {"name": "Yitzhak Navon", "lat": 31.7881, "lon": 35.2028, "passengers": 15000},
]


def load_data():
    areas = gpd.read_file(script_dir / "jer_areas.shp")
    areas = areas[areas['in_jeru'] == 1].copy()
    # Keep population and employment for all years
    for year in DATA_YEARS:
        areas[f'pop_{year}'] = areas[f'pop_{year}'].fillna(0)
        areas[f'emp_{year}'] = areas[f'emp_{year}'].fillna(0)
    # Default year for backwards compatibility
    areas['pop'] = areas[f'pop_{DEFAULT_YEAR}'].fillna(0)
    areas['emp'] = areas[f'emp_{DEFAULT_YEAR}'].fillna(0)

    roads = gpd.read_file(script_dir / "jerusalem_roads.kml", driver='KML')
    completed = gpd.read_file(script_dir / "bike_lanes_completed.kml", driver='KML')
    construction = gpd.read_file(script_dir / "bike_lanes_construction.kml", driver='KML')
    try:
        plan = gpd.read_file(script_dir / "bike_lanes_plan.kml", driver='KML', on_invalid='ignore')
        # Drop any features with None geometry (from invalid WKB)
        plan = plan[plan.geometry.notnull()].copy()
    except Exception as e:
        print(f"Warning: Could not read bike_lanes_plan.kml ({e}), using empty layer")
        plan = gpd.GeoDataFrame(columns=['geometry', 'Name'], geometry='geometry', crs='EPSG:4326')
    try:
        check = gpd.read_file(script_dir / "bike_lanes_check.kml", driver='KML', on_invalid='ignore')
        check = check[check.geometry.notnull()].copy()
    except Exception as e:
        print(f"Warning: Could not read bike_lanes_check.kml ({e}), using empty layer")
        check = gpd.GeoDataFrame(columns=['geometry', 'Name'], geometry='geometry', crs='EPSG:4326')
    wishing = gpd.read_file(script_dir / "bike_lanes_wishing_list.kml", driver='KML')

    return areas, roads, completed, construction, plan, check, wishing


def load_highway_types():
    """Load OSM highway classification for roads and return name->discomfort mapping."""
    csv_path = script_dir / "jerusalem_roads_major.csv"
    if not csv_path.exists():
        print("Warning: jerusalem_roads_major.csv not found, using default discomfort=1.0 for all roads")
        return {}
    import csv
    name_to_discomfort = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get('name:he') or row.get('name', '')
            highway = row.get('highway', 'residential')
            if name:
                discomfort = DISCOMFORT_FACTORS.get(highway, 1.0)
                # Keep the highest discomfort if a road appears multiple times
                if name not in name_to_discomfort or discomfort > name_to_discomfort[name]:
                    name_to_discomfort[name] = discomfort
    return name_to_discomfort


def assign_destination_layers(areas_proj):
    """Assign education and transit destination values to statistical areas.

    Each institution/station is assigned to the area polygon it falls in,
    or the nearest area centroid if outside all polygons.
    """
    from shapely.geometry import Point

    transformer = Transformer.from_crs(WGS84, TARGET_CRS, always_xy=True)

    areas_proj['students'] = 0.0
    areas_proj['transit'] = 0.0

    centroids = areas_proj.geometry.centroid

    def find_area_for_point(lon, lat):
        """Find which area a point falls in, or nearest centroid."""
        x, y = transformer.transform(lon, lat)
        pt = Point(x, y)
        # Try point-in-polygon first
        for idx, row in areas_proj.iterrows():
            if row.geometry.contains(pt):
                return idx
        # Fallback: nearest centroid
        min_dist = float('inf')
        best_idx = areas_proj.index[0]
        for idx, c in zip(areas_proj.index, centroids):
            d = pt.distance(c)
            if d < min_dist:
                min_dist = d
                best_idx = idx
        return best_idx

    for inst in INSTITUTIONS:
        idx = find_area_for_point(inst['lon'], inst['lat'])
        areas_proj.loc[idx, 'students'] += inst['students']

    for station in TRAIN_STATIONS:
        idx = find_area_for_point(station['lon'], station['lat'])
        areas_proj.loc[idx, 'transit'] += station['passengers']

    n_areas_with_students = (areas_proj['students'] > 0).sum()
    n_areas_with_transit = (areas_proj['transit'] > 0).sum()
    total_students = areas_proj['students'].sum()
    total_transit = areas_proj['transit'].sum()
    print(f"  Education: {total_students:.0f} students in {n_areas_with_students} areas")
    print(f"  Transit: {total_transit:.0f} daily passengers in {n_areas_with_transit} areas")


def build_network(roads_proj, bike_lanes_list, areas_proj=None, tolerance=NODE_TOLERANCE, road_discomfort=None):
    from shapely.strtree import STRtree
    from shapely.geometry import Point

    if road_discomfort is None:
        road_discomfort = {}

    G = nx.Graph()
    coord_to_node = {}
    node_coords = {}
    node_counter = [0]
    edge_to_geom = {}  # (s, e) -> road geometry for spatial matching

    def get_or_create_node(x, y):
        key = (round(x / tolerance) * tolerance, round(y / tolerance) * tolerance)
        if key in coord_to_node:
            return coord_to_node[key]
        nid = node_counter[0]
        node_counter[0] += 1
        coord_to_node[key] = nid
        node_coords[nid] = (x, y)
        return nid

    # First, add all road segments and keep track of geometries
    road_geoms = []
    road_edges = []
    for _, row in roads_proj.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or geom.geom_type != 'LineString':
            continue
        coords = list(geom.coords)
        if len(coords) >= 2:
            s = get_or_create_node(coords[0][0], coords[0][1])
            e = get_or_create_node(coords[-1][0], coords[-1][1])
            if s != e:
                road_name = row.get('Name', '') or ''
                discomfort = road_discomfort.get(road_name, 1.0)
                G.add_edge(s, e, length=geom.length, has_bike_lane=False, discomfort=discomfort)
                edge_key = (min(s, e), max(s, e))
                edge_to_geom[edge_key] = geom
                road_geoms.append(geom)
                road_edges.append(edge_key)

    # Build spatial index for road geometries
    road_tree_spatial = STRtree(road_geoms) if road_geoms else None

    # Mark road edges that have bike lanes running along them
    BUFFER_DIST = 25  # meters - bike lane must be within 25m of road (increased from 15m)

    def mark_bike_lane_roads(line_geom):
        """Find and mark all road edges that this bike lane runs along."""
        if road_tree_spatial is None:
            return
        # Buffer the bike lane to find nearby roads
        buffered = line_geom.buffer(BUFFER_DIST)
        # Find candidate road geometries
        candidate_indices = road_tree_spatial.query(buffered)
        for idx in candidate_indices:
            road_geom = road_geoms[idx]
            # Check if the road segment significantly overlaps with the bike lane
            # Use intersection length as a measure
            try:
                intersection = road_geom.intersection(buffered)
                if intersection.is_empty:
                    continue
                # Calculate overlap ratio
                overlap_ratio = intersection.length / road_geom.length if road_geom.length > 0 else 0

                # Also check the reverse: does the bike lane pass through the road's buffer?
                # This helps catch parallel roads that run alongside bike lanes
                road_buffered = road_geom.buffer(BUFFER_DIST)
                bike_in_road_buffer = line_geom.intersection(road_buffered)
                bike_overlap = bike_in_road_buffer.length / line_geom.length if line_geom.length > 0 else 0

                # Mark as bike lane if:
                # 1. At least 50% of road segment is covered by bike lane, OR
                # 2. Road segment is short (<50m) and at least 30% is covered, OR
                # 3. The bike lane covers a significant absolute length (>20m) of the road, OR
                # 4. The road and bike lane are parallel (both have significant mutual coverage)
                should_mark = (
                    overlap_ratio > 0.5 or
                    (road_geom.length < 50 and overlap_ratio > 0.3) or
                    intersection.length > 20 or
                    (overlap_ratio > 0.3 and bike_overlap > 0.1)  # Parallel road detection
                )

                if should_mark:
                    edge_key = road_edges[idx]
                    s, e = edge_key
                    if G.has_edge(s, e):
                        G[s][e]['has_bike_lane'] = True
            except:
                pass

    for bl_gdf in bike_lanes_list:
        if bl_gdf is None or len(bl_gdf) == 0:
            continue
        bl_proj = bl_gdf.to_crs(TARGET_CRS)
        for _, row in bl_proj.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            # Handle both LineString and MultiLineString
            if geom.geom_type == 'LineString':
                mark_bike_lane_roads(geom)
            elif geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    mark_bike_lane_roads(line)

    # Create virtual edges for bike lanes that don't follow roads
    # This allows dedicated bike paths (through parks, along rivers, etc.) to be used for routing
    VIRTUAL_EDGE_THRESHOLD = 50  # meters - nodes within this distance can connect to the bike lane

    def create_virtual_edges_for_lane(line_geom):
        """Create virtual edges connecting network nodes along a bike lane."""
        from shapely.ops import substring

        if len(node_coords) == 0:
            return 0

        # Build KD-tree of current nodes if we have enough
        current_node_ids = list(node_coords.keys())
        current_coords = np.array([node_coords[n] for n in current_node_ids])
        if len(current_coords) < 2:
            return 0
        current_tree = cKDTree(current_coords)

        # Find all nodes within threshold of any point on the line
        # Sample points along the line
        line_length = line_geom.length
        if line_length < 10:  # Skip very short segments
            return 0

        # Find nodes near the line
        nearby_nodes = []
        line_buffer = line_geom.buffer(VIRTUAL_EDGE_THRESHOLD)

        for idx, nid in enumerate(current_node_ids):
            pt = Point(current_coords[idx])
            if line_buffer.contains(pt):
                # Project node onto line to get distance along
                proj_dist = line_geom.project(pt)
                perp_dist = pt.distance(line_geom)
                if perp_dist <= VIRTUAL_EDGE_THRESHOLD:
                    nearby_nodes.append({
                        'node_id': nid,
                        'proj_dist': proj_dist,
                        'perp_dist': perp_dist
                    })

        # Sort by distance along the line
        nearby_nodes.sort(key=lambda x: x['proj_dist'])

        # Create virtual edges between consecutive nodes
        virtual_edge_count = 0
        for i in range(len(nearby_nodes) - 1):
            n1 = nearby_nodes[i]
            n2 = nearby_nodes[i + 1]

            # Edge length is distance along the bike lane between projections
            edge_len = n2['proj_dist'] - n1['proj_dist']

            # Only create edge if there's meaningful distance
            if edge_len > 5:  # At least 5 meters
                node_a, node_b = n1['node_id'], n2['node_id']
                # Check if this edge already exists
                if not G.has_edge(node_a, node_b):
                    G.add_edge(node_a, node_b, length=edge_len, has_bike_lane=True, is_virtual=True)
                    virtual_edge_count += 1
                    # Extract and store the geometry for this virtual edge
                    try:
                        edge_geom = substring(line_geom, n1['proj_dist'], n2['proj_dist'])
                        if edge_geom and not edge_geom.is_empty and edge_geom.geom_type == 'LineString':
                            edge_key = (min(node_a, node_b), max(node_a, node_b))
                            edge_to_geom[edge_key] = edge_geom
                    except:
                        pass  # If geometry extraction fails, edge still works but renders as straight line
                elif not G[node_a][node_b].get('has_bike_lane'):
                    # Edge exists but wasn't marked as bike lane - update it
                    G[node_a][node_b]['has_bike_lane'] = True

        return virtual_edge_count

    total_virtual_edges = 0
    for bl_gdf in bike_lanes_list:
        if bl_gdf is None or len(bl_gdf) == 0:
            continue
        bl_proj = bl_gdf.to_crs(TARGET_CRS)
        for _, row in bl_proj.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == 'LineString':
                count = create_virtual_edges_for_lane(geom)
                if count:
                    total_virtual_edges += count
            elif geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    count = create_virtual_edges_for_lane(line)
                    if count:
                        total_virtual_edges += count

    if total_virtual_edges > 0:
        print(f"  Created {total_virtual_edges} virtual edges for dedicated bike paths")

    # Connect off-road bike lanes to the road network
    # For lanes that are far from roads (parks, trails), create:
    # 1. Nodes along the bike lane
    # 2. Connector edges to nearest road nodes
    CONNECTOR_THRESHOLD = 1000  # Max distance to create connector to road network
    LANE_NODE_INTERVAL = 100    # Create nodes every 100m along off-road lanes

    def connect_offroad_lane(line_geom):
        """Connect an off-road bike lane to the road network."""
        if line_geom.length < 20:
            return 0

        # Build current node tree
        current_node_ids = list(node_coords.keys())
        if len(current_node_ids) < 2:
            return 0
        current_coords = np.array([node_coords[n] for n in current_node_ids])
        current_tree = cKDTree(current_coords)

        # Check if lane endpoints are far from road network
        start = Point(line_geom.coords[0])
        end = Point(line_geom.coords[-1])

        start_dist, start_idx = current_tree.query([start.x, start.y])
        end_dist, end_idx = current_tree.query([end.x, end.y])

        # If both endpoints are already within virtual threshold, standard logic handles it
        if start_dist <= VIRTUAL_EDGE_THRESHOLD and end_dist <= VIRTUAL_EDGE_THRESHOLD:
            return 0

        connections_made = 0

        # Create nodes at lane endpoints if they're far from existing nodes
        if start_dist > VIRTUAL_EDGE_THRESHOLD and start_dist <= CONNECTOR_THRESHOLD:
            start_node = get_or_create_node(start.x, start.y)
            nearest_road_node = current_node_ids[start_idx]
            if start_node != nearest_road_node and not G.has_edge(start_node, nearest_road_node):
                # Connector edge - allows access but not a bike lane itself
                G.add_edge(start_node, nearest_road_node,
                          length=start_dist,
                          has_bike_lane=False,
                          is_connector=True)
                connections_made += 1

        if end_dist > VIRTUAL_EDGE_THRESHOLD and end_dist <= CONNECTOR_THRESHOLD:
            end_node = get_or_create_node(end.x, end.y)
            # Re-query since we may have added start_node
            current_node_ids_updated = list(node_coords.keys())
            current_coords_updated = np.array([node_coords[n] for n in current_node_ids_updated])
            current_tree_updated = cKDTree(current_coords_updated)
            end_dist_new, end_idx_new = current_tree_updated.query([end.x, end.y])

            if end_dist_new > VIRTUAL_EDGE_THRESHOLD:
                nearest_road_node = current_node_ids_updated[end_idx_new]
                if end_node != nearest_road_node and not G.has_edge(end_node, nearest_road_node):
                    G.add_edge(end_node, nearest_road_node,
                              length=end_dist_new,
                              has_bike_lane=False,
                              is_connector=True)
                    connections_made += 1

        # Create nodes and edges along the bike lane itself
        if line_geom.length > LANE_NODE_INTERVAL:
            num_segments = max(2, int(line_geom.length / LANE_NODE_INTERVAL))
            prev_node = None

            for i in range(num_segments + 1):
                t = i / num_segments
                pt = line_geom.interpolate(t, normalized=True)
                node = get_or_create_node(pt.x, pt.y)

                if prev_node is not None and prev_node != node:
                    segment_length = line_geom.length / num_segments
                    if not G.has_edge(prev_node, node):
                        G.add_edge(prev_node, node,
                                  length=segment_length,
                                  has_bike_lane=True,
                                  is_virtual=True)
                        connections_made += 1

                prev_node = node
        else:
            # Short lane - just connect endpoints
            start_node = get_or_create_node(start.x, start.y)
            end_node = get_or_create_node(end.x, end.y)
            if start_node != end_node and not G.has_edge(start_node, end_node):
                G.add_edge(start_node, end_node,
                          length=line_geom.length,
                          has_bike_lane=True,
                          is_virtual=True)
                connections_made += 1

        return connections_made

    # Process all bike lanes for off-road connections
    offroad_connections = 0
    offroad_lanes = 0
    for bl_gdf in bike_lanes_list:
        if bl_gdf is None or len(bl_gdf) == 0:
            continue
        bl_proj = bl_gdf.to_crs(TARGET_CRS)
        for _, row in bl_proj.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == 'LineString':
                count = connect_offroad_lane(geom)
                if count > 0:
                    offroad_connections += count
                    offroad_lanes += 1
            elif geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    count = connect_offroad_lane(line)
                    if count > 0:
                        offroad_connections += count
                        offroad_lanes += 1

    if offroad_connections > 0:
        print(f"  Connected {offroad_lanes} off-road bike lanes with {offroad_connections} edges")

    # Connect area centroids to the nearest roads
    # This ensures every area has a proper connection to the network
    if areas_proj is not None and len(road_geoms) > 0:
        road_tree_for_areas = STRtree(road_geoms)

        for _, area_row in areas_proj.iterrows():
            centroid = area_row.geometry.centroid
            centroid_pt = Point(centroid.x, centroid.y)

            # Find nearest road geometry
            nearest_idx = road_tree_for_areas.nearest(centroid_pt)
            nearest_road = road_geoms[nearest_idx]

            # Find the nearest point on that road
            nearest_point_on_road = nearest_road.interpolate(nearest_road.project(centroid_pt))
            dist_to_road = centroid_pt.distance(nearest_point_on_road)

            # Create a node at the nearest point on the road
            road_node = get_or_create_node(nearest_point_on_road.x, nearest_point_on_road.y)

            # Get the edge that this road corresponds to
            edge_key = road_edges[nearest_idx]
            s, e = edge_key

            # If the road node is different from both endpoints, we need to split the edge
            if road_node != s and road_node != e and G.has_edge(s, e):
                old_length = G[s][e]['length']
                old_has_bike = G[s][e].get('has_bike_lane', False)

                # Calculate distances from road node to both endpoints
                s_coord = node_coords[s]
                e_coord = node_coords[e]
                road_node_coord = node_coords[road_node]

                dist_to_s = np.sqrt((road_node_coord[0] - s_coord[0])**2 + (road_node_coord[1] - s_coord[1])**2)
                dist_to_e = np.sqrt((road_node_coord[0] - e_coord[0])**2 + (road_node_coord[1] - e_coord[1])**2)

                # Only split if the new node is meaningfully inside the edge
                if dist_to_s > tolerance and dist_to_e > tolerance:
                    # Remove old edge
                    G.remove_edge(s, e)
                    # Add two new edges
                    G.add_edge(s, road_node, length=dist_to_s, has_bike_lane=old_has_bike)
                    G.add_edge(road_node, e, length=dist_to_e, has_bike_lane=old_has_bike)
                    # Update edge_to_geom for the new edges
                    new_key_s = (min(s, road_node), max(s, road_node))
                    new_key_e = (min(road_node, e), max(road_node, e))
                    # Keep reference to original geometry for path drawing
                    if edge_key in edge_to_geom:
                        edge_to_geom[new_key_s] = edge_to_geom[edge_key]
                        edge_to_geom[new_key_e] = edge_to_geom[edge_key]

            # If centroid is significantly far from the road, add a connector edge
            # This creates a direct path from area to road network
            if dist_to_road > tolerance:
                centroid_node = get_or_create_node(centroid.x, centroid.y)
                if centroid_node != road_node:
                    # Add edge connecting centroid to road network
                    G.add_edge(centroid_node, road_node, length=dist_to_road, has_bike_lane=False)

    node_ids = list(node_coords.keys())
    coords_array = np.array([node_coords[n] for n in node_ids])
    node_tree = cKDTree(coords_array) if len(coords_array) > 0 else None

    # Ensure all areas are connected to the largest component
    # Some areas might be near disconnected road segments
    if areas_proj is not None and node_tree is not None:
        largest_cc = max(nx.connected_components(G), key=len)
        # Build a KD-tree of only the nodes in the largest component
        cc_nodes = [n for n in node_ids if n in largest_cc]
        if cc_nodes:
            cc_coords = np.array([node_coords[n] for n in cc_nodes])
            cc_tree = cKDTree(cc_coords)

            for _, area_row in areas_proj.iterrows():
                centroid = area_row.geometry.centroid
                # Find the nearest node to this centroid
                _, nearest_idx = node_tree.query([centroid.x, centroid.y])
                nearest_node = node_ids[nearest_idx]

                # If it's not in the largest component, connect it
                if nearest_node not in largest_cc:
                    # Find the nearest node in the largest component
                    dist, cc_idx = cc_tree.query([centroid.x, centroid.y])
                    cc_node = cc_nodes[cc_idx]

                    # Add edge from the nearest node to the main network
                    if nearest_node != cc_node:
                        nearest_coord = node_coords[nearest_node]
                        cc_coord = node_coords[cc_node]
                        edge_length = np.sqrt((nearest_coord[0] - cc_coord[0])**2 +
                                             (nearest_coord[1] - cc_coord[1])**2)
                        G.add_edge(nearest_node, cc_node, length=edge_length, has_bike_lane=False)
                        # Update the largest component (now includes the connected nodes)
                        largest_cc = max(nx.connected_components(G), key=len)

    return G, node_coords, node_tree, node_ids, edge_to_geom


def compute_area_accessibility(G, node_coords, node_tree, node_ids, areas_proj, theta, k,
                                w_emp=1.0, w_edu=0.0, w_transit=0.0):
    """Compute origin and destination accessibility for each area.
    origin: acc_orig[i] = sum_j dest_weight_j * tau_ij^theta
    dest:   acc_dest[j] = sum_i P_i * tau_ij^theta
    where dest_weight_j = w_emp*E_j + w_edu*S_j + w_transit*T_j
    """
    Gw = G.copy()
    for u, v in Gw.edges():
        l = Gw[u][v]['length']
        discomfort = Gw[u][v].get('discomfort', 1.0)
        Gw[u][v]['weight'] = l if Gw[u][v].get('has_bike_lane') else l * k * discomfort

    centroids = areas_proj.geometry.centroid
    n = len(areas_proj)
    center_nodes = [node_ids[node_tree.query([c.x, c.y])[1]] for c in centroids]
    pop = areas_proj['pop'].values
    emp = areas_proj['emp'].values
    students = areas_proj['students'].values if 'students' in areas_proj.columns else np.zeros(n)
    transit = areas_proj['transit'].values if 'transit' in areas_proj.columns else np.zeros(n)
    largest_cc = max(nx.connected_components(Gw), key=len)

    acc_orig = np.zeros(n)
    acc_dest = np.zeros(n)
    for i in range(n):
        if center_nodes[i] not in largest_cc:
            continue
        try:
            dists = nx.single_source_dijkstra_path_length(Gw, center_nodes[i], weight='weight')
        except Exception:
            continue
        for j in range(n):
            if i != j and center_nodes[j] in dists:
                tau = max(dists[center_nodes[j]] / 1000, 0.1)
                decay = tau ** theta
                dest_w = w_emp * emp[j] + w_edu * students[j] + w_transit * transit[j]
                acc_orig[i] += dest_w * decay
                acc_dest[j] += pop[i] * decay
    return acc_orig, acc_dest


def geojson_from_gdf(gdf, props_list):
    """Convert GeoDataFrame to GeoJSON dict, only keeping specified properties."""
    gdf_wgs = gdf.to_crs(WGS84)
    features = []
    for idx, row in gdf_wgs.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        props = {}
        for p in props_list:
            val = row.get(p, None)
            if val is not None:
                if isinstance(val, (np.integer,)):
                    val = int(val)
                elif isinstance(val, (np.floating,)):
                    val = round(float(val), 2)
                props[p] = val
        features.append({
            "type": "Feature",
            "geometry": json.loads(gdf_wgs.loc[[idx]].geometry.to_json())["features"][0]["geometry"],
            "properties": props
        })
    return {"type": "FeatureCollection", "features": features}


def main():
    print("Loading data...")
    areas, roads, completed, construction, plan, check, wishing = load_data()

    areas_proj = areas.to_crs(TARGET_CRS)
    roads_proj = roads.to_crs(TARGET_CRS)

    # Load road discomfort factors from OSM highway classification
    print("Loading road discomfort factors...")
    road_discomfort = load_highway_types()
    print(f"  Loaded discomfort factors for {len(road_discomfort)} road names")

    # Assign education and transit destinations to areas
    print("Assigning destination layers...")
    assign_destination_layers(areas_proj)
    # Copy back to areas for GeoJSON export
    areas['students'] = areas_proj['students'].values
    areas['transit'] = areas_proj['transit'].values

    # Prepare area names and IDs
    areas['area_id'] = range(len(areas))
    areas_proj['area_id'] = range(len(areas_proj))
    area_names = []
    for _, row in areas.iterrows():
        name = row.get('name', None)
        if name is None or (isinstance(name, float) and np.isnan(name)):
            name = f"Area {row['area_id']}"
        area_names.append(str(name))

    # Build GeoJSON for display
    print("Building GeoJSON layers...")
    areas_geojson = geojson_from_gdf(areas[['geometry', 'pop', 'emp', 'students', 'transit', 'area_id']], ['pop', 'emp', 'students', 'transit', 'area_id'])
    # attach area_id to features properly
    for i, feat in enumerate(areas_geojson['features']):
        feat['properties']['area_id'] = i

    completed_geojson = geojson_from_gdf(completed[['geometry', 'Name']], ['Name']) if len(completed) > 0 else {"type": "FeatureCollection", "features": []}
    construction_geojson = geojson_from_gdf(construction[['geometry', 'Name']], ['Name']) if len(construction) > 0 else {"type": "FeatureCollection", "features": []}
    plan_geojson = geojson_from_gdf(plan[['geometry', 'Name']], ['Name']) if len(plan) > 0 else {"type": "FeatureCollection", "features": []}
    check_geojson = geojson_from_gdf(check[['geometry', 'Name']], ['Name']) if len(check) > 0 else {"type": "FeatureCollection", "features": []}

    # Wishing list - use integer lane_id for identification
    wishing['lane_id'] = range(len(wishing))
    wishing_geojson = geojson_from_gdf(wishing[['geometry', 'Name', 'lane_id']], ['Name', 'lane_id'])
    lane_names = [wishing.iloc[i]['Name'] for i in range(len(wishing))]

    # Compute lane lengths in meters (using projected CRS)
    wishing_proj_temp = wishing.to_crs(TARGET_CRS)
    lane_lengths = [round(wishing_proj_temp.iloc[i].geometry.length, 1) if wishing_proj_temp.iloc[i].geometry else 0 for i in range(len(wishing_proj_temp))]

    # Build network with ALL lane types for proper node/edge topology
    # All lanes contribute to the base graph structure (nodes, edges, virtual edges)
    # The JS side decides which layers are "active" for bike-lane weighting
    print("Building network...")
    all_lane_layers = [completed, construction, plan, check, wishing]
    G_base, nc, nt, ni, edge_geoms = build_network(roads_proj, all_lane_layers, areas_proj, road_discomfort=road_discomfort)
    print(f"  Network: {G_base.number_of_nodes()} nodes, {G_base.number_of_edges()} edges")

    # Build network data for path finding (WGS84 coords)
    # EDGES are exported with has_bike_lane=0 for all edges (clean base network)
    # Each layer type has its own edge set computed below; JS combines active layers
    print("Exporting network for path finding...")
    transformer = Transformer.from_crs(TARGET_CRS, WGS84, always_xy=True)
    nodes_wgs = {}
    for nid, (x, y) in nc.items():
        lon, lat = transformer.transform(x, y)
        nodes_wgs[str(nid)] = [round(lon, 6), round(lat, 6)]

    edges_list = []
    for u, v, d in G_base.edges(data=True):
        # All edges start as has_bike_lane=0; JS toggles via layer edge sets
        # 5th element is discomfort factor for road type
        discomfort = round(d.get('discomfort', 1.0), 1)
        edges_list.append([u, v, round(d['length'], 1), 0, discomfort])

    # Export edge geometries for accurate path drawing
    edge_geoms_wgs = {}
    for (s, e), geom in edge_geoms.items():
        # Transform geometry coords to WGS84
        coords_wgs = []
        for coord in geom.coords:
            # Handle both 2D (x, y) and 3D (x, y, z) coordinates
            x, y = coord[0], coord[1]
            lon, lat = transformer.transform(x, y)
            coords_wgs.append([round(lon, 6), round(lat, 6)])
        edge_key = f"{min(s,e)}_{max(s,e)}"
        edge_geoms_wgs[edge_key] = coords_wgs

    # Pre-compute edges for each layer type (for online path calculation)
    # Find which road edges each layer covers (same approach as build_network)
    print("Computing layer edges for online path finding...")
    from shapely.strtree import STRtree

    wishing_proj = wishing.to_crs(TARGET_CRS)
    wishing_wgs = wishing.to_crs(WGS84)
    wishing_edges = {}  # lane_id -> [[nodeA, nodeB], ...] - road edges this lane covers
    wishing_geoms = {}  # lane_id -> [[lon, lat], ...] for visualization

    # Build road geometry index from roads_proj (need to rebuild for wishing lane matching)
    road_geoms_list = []
    road_edges_list = []
    for _, row in roads_proj.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or geom.geom_type != 'LineString':
            continue
        coords = list(geom.coords)
        if len(coords) >= 2:
            # Find the nodes for this road segment
            sx, sy = coords[0][0], coords[0][1]
            ex, ey = coords[-1][0], coords[-1][1]
            _, s_idx = nt.query([sx, sy])
            _, e_idx = nt.query([ex, ey])
            s, e = ni[s_idx], ni[e_idx]
            if s != e:
                edge_key = (min(s, e), max(s, e))
                road_geoms_list.append(geom)
                road_edges_list.append(edge_key)

    road_tree_spatial = STRtree(road_geoms_list) if road_geoms_list else None
    BUFFER_DIST = 25  # meters (increased from 15m to better catch parallel roads)

    def get_linestrings(geom):
        """Extract LineStrings from any geometry type."""
        if geom.geom_type == 'LineString':
            return [geom]
        elif geom.geom_type == 'MultiLineString':
            return list(geom.geoms)
        return []

    def check_should_mark(line, road_geom, buffered):
        """Check if a road should be marked as having a bike lane."""
        intersection = road_geom.intersection(buffered)
        if intersection.is_empty:
            return False
        overlap_ratio = intersection.length / road_geom.length if road_geom.length > 0 else 0

        # Also check the reverse: does the bike lane pass through the road's buffer?
        road_buffered = road_geom.buffer(BUFFER_DIST)
        bike_in_road_buffer = line.intersection(road_buffered)
        bike_overlap = bike_in_road_buffer.length / line.length if line.length > 0 else 0

        # Mark as bike lane if:
        # 1. At least 50% of road segment is covered by bike lane, OR
        # 2. Road segment is short (<50m) and at least 30% is covered, OR
        # 3. The bike lane covers a significant absolute length (>20m) of the road, OR
        # 4. The road and bike lane are parallel (both have significant mutual coverage)
        return (
            overlap_ratio > 0.5 or
            (road_geom.length < 50 and overlap_ratio > 0.3) or
            intersection.length > 20 or
            (overlap_ratio > 0.3 and bike_overlap > 0.1)  # Parallel road detection
        )

    for lid in range(len(wishing_proj)):
        geom = wishing_proj.iloc[lid].geometry
        geom_wgs = wishing_wgs.iloc[lid].geometry
        if geom is None or geom.is_empty:
            wishing_edges[lid] = []
            wishing_geoms[lid] = []
            continue

        # Store WGS84 coordinates for visualization
        lines_wgs = get_linestrings(geom_wgs)
        all_coords = []
        for line in lines_wgs:
            all_coords.extend([[round(c[0], 6), round(c[1], 6)] for c in line.coords])
        wishing_geoms[lid] = all_coords if all_coords else []

        # Find road edges this wishing lane covers
        covered_edges = set()
        for line in get_linestrings(geom):
            if road_tree_spatial is None:
                continue
            buffered = line.buffer(BUFFER_DIST)
            candidate_indices = road_tree_spatial.query(buffered)
            for idx in candidate_indices:
                road_geom = road_geoms_list[idx]
                try:
                    if check_should_mark(line, road_geom, buffered):
                        covered_edges.add(road_edges_list[idx])
                except:
                    pass

        # Store as list of [nodeA, nodeB] pairs
        wishing_edges[lid] = [[e[0], e[1]] for e in covered_edges]

    # Compute edges for each non-wishing layer type (completed, construction, plan, check)
    # These are computed as a single set of edges per layer (not per-lane)
    def compute_layer_edges(layer_gdf):
        """Find which road edges a layer covers (same logic as wishing lanes)."""
        if layer_gdf is None or len(layer_gdf) == 0 or road_tree_spatial is None:
            return []
        layer_proj = layer_gdf.to_crs(TARGET_CRS)
        covered = set()
        for _, row in layer_proj.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            for line in get_linestrings(geom):
                buffered = line.buffer(BUFFER_DIST)
                candidate_indices = road_tree_spatial.query(buffered)
                for idx in candidate_indices:
                    road_geom = road_geoms_list[idx]
                    try:
                        if check_should_mark(line, road_geom, buffered):
                            covered.add(road_edges_list[idx])
                    except:
                        pass
        return [[e[0], e[1]] for e in covered]

    print("  Computing layer edges...")
    completed_edges = compute_layer_edges(completed)
    construction_edges = compute_layer_edges(construction)
    plan_edges = compute_layer_edges(plan)
    check_edges = compute_layer_edges(check)
    print(f"  Completed: {len(completed_edges)} edges, Construction: {len(construction_edges)} edges, Plan: {len(plan_edges)} edges, Check: {len(check_edges)} edges")

    # Compute virtual edges for each non-wishing layer type
    def compute_layer_virtual_edges(layer_gdf):
        """Create virtual edges for a layer (same logic as wishing lanes)."""
        if layer_gdf is None or len(layer_gdf) == 0 or node_tree is None:
            return []
        from shapely.ops import substring as substr
        from shapely.geometry import Point as Pt
        layer_proj_data = layer_gdf.to_crs(TARGET_CRS)
        layer_wgs_data = layer_gdf.to_crs(WGS84)
        virtual_edges_result = []
        for row_idx in range(len(layer_proj_data)):
            geom = layer_proj_data.iloc[row_idx].geometry
            geom_wgs = layer_wgs_data.iloc[row_idx].geometry
            if geom is None or geom.is_empty:
                continue
            for line, line_wgs in zip(get_linestrings(geom), get_linestrings(geom_wgs)):
                line_length = line.length
                if line_length < 10:
                    continue
                nearby = []
                line_buffer = line.buffer(VIRTUAL_EDGE_THRESHOLD)
                for idx_n, nid in enumerate(ni_list):
                    pt = Pt(nc_list[idx_n])
                    if line_buffer.contains(pt):
                        proj_dist = line.project(pt)
                        perp_dist = pt.distance(line)
                        if perp_dist <= VIRTUAL_EDGE_THRESHOLD:
                            nearby.append({'node_id': nid, 'proj_dist': proj_dist, 'perp_dist': perp_dist})
                nearby.sort(key=lambda x: x['proj_dist'])
                for i in range(len(nearby) - 1):
                    n1 = nearby[i]
                    n2 = nearby[i + 1]
                    edge_len = n2['proj_dist'] - n1['proj_dist']
                    if edge_len > 5:
                        try:
                            edge_geom_wgs = substr(line_wgs, n1['proj_dist'], n2['proj_dist'])
                            if edge_geom_wgs and not edge_geom_wgs.is_empty and edge_geom_wgs.geom_type == 'LineString':
                                coords_wgs = [[round(c[0], 6), round(c[1], 6)] for c in edge_geom_wgs.coords]
                                virtual_edges_result.append({
                                    'from': n1['node_id'], 'to': n2['node_id'],
                                    'len': round(edge_len, 1), 'geometry': coords_wgs
                                })
                        except:
                            virtual_edges_result.append({
                                'from': n1['node_id'], 'to': n2['node_id'],
                                'len': round(edge_len, 1)
                            })
        return virtual_edges_result

    # Create virtual edges for wishing lanes (similar to existing bike lanes)
    # This ensures wishing lanes can provide connectivity even where they don't follow roads
    print("Creating virtual edges for wishing lanes...")
    from shapely.ops import substring
    from shapely.geometry import Point

    VIRTUAL_EDGE_THRESHOLD = 50  # meters - nodes within this distance can connect
    wishing_virtual_edges = {}  # lane_id -> [{from, to, len, geometry}, ...]

    # Build node tree for finding nearby nodes
    ni_list = list(ni)
    nc_list = [nc[n] for n in ni_list]
    if len(nc_list) > 0:
        node_tree = cKDTree(nc_list)
    else:
        node_tree = None

    for lid in range(len(wishing_proj)):
        geom = wishing_proj.iloc[lid].geometry
        geom_wgs = wishing_wgs.iloc[lid].geometry
        if geom is None or geom.is_empty or node_tree is None:
            wishing_virtual_edges[lid] = []
            continue

        virtual_edges = []
        for line, line_wgs in zip(get_linestrings(geom), get_linestrings(geom_wgs)):
            line_length = line.length
            if line_length < 10:
                continue

            # Find nodes near this line
            nearby_nodes = []
            line_buffer = line.buffer(VIRTUAL_EDGE_THRESHOLD)

            for idx, nid in enumerate(ni_list):
                pt = Point(nc_list[idx])
                if line_buffer.contains(pt):
                    proj_dist = line.project(pt)
                    perp_dist = pt.distance(line)
                    if perp_dist <= VIRTUAL_EDGE_THRESHOLD:
                        nearby_nodes.append({
                            'node_id': nid,
                            'proj_dist': proj_dist,
                            'perp_dist': perp_dist
                        })

            nearby_nodes.sort(key=lambda x: x['proj_dist'])

            # Create virtual edges between consecutive nodes
            for i in range(len(nearby_nodes) - 1):
                n1 = nearby_nodes[i]
                n2 = nearby_nodes[i + 1]
                edge_len = n2['proj_dist'] - n1['proj_dist']

                if edge_len > 5:  # At least 5 meters
                    # Extract geometry portion in WGS84
                    try:
                        edge_geom = substring(line, n1['proj_dist'], n2['proj_dist'])
                        edge_geom_wgs = substring(line_wgs, n1['proj_dist'], n2['proj_dist'])
                        if edge_geom_wgs and not edge_geom_wgs.is_empty and edge_geom_wgs.geom_type == 'LineString':
                            coords_wgs = [[round(c[0], 6), round(c[1], 6)] for c in edge_geom_wgs.coords]
                            virtual_edges.append({
                                'from': n1['node_id'],
                                'to': n2['node_id'],
                                'len': round(edge_len, 1),
                                'geometry': coords_wgs
                            })
                    except:
                        # Fall back to straight line
                        virtual_edges.append({
                            'from': n1['node_id'],
                            'to': n2['node_id'],
                            'len': round(edge_len, 1)
                        })

        wishing_virtual_edges[lid] = virtual_edges

    total_wishing_virtual = sum(len(v) for v in wishing_virtual_edges.values())
    print(f"  Created {total_wishing_virtual} virtual edges for wishing lanes")

    # Compute virtual edges for non-wishing layer types
    print("Computing virtual edges for other layer types...")
    completed_virtual_edges = compute_layer_virtual_edges(completed)
    construction_virtual_edges = compute_layer_virtual_edges(construction)
    plan_virtual_edges = compute_layer_virtual_edges(plan)
    check_virtual_edges = compute_layer_virtual_edges(check)
    print(f"  Completed: {len(completed_virtual_edges)}, Construction: {len(construction_virtual_edges)}, Plan: {len(plan_virtual_edges)}, Check: {len(check_virtual_edges)} virtual edges")

    # Area centroids in WGS84
    areas_wgs = areas.to_crs(WGS84)
    centroids_wgs = [[round(c.x, 6), round(c.y, 6)] for c in areas_wgs.geometry.centroid]

    # Generate HTML
    print("Generating HTML...")
    # Extract area data for online computation - all years
    area_pop_by_year = {}
    area_emp_by_year = {}
    for year in DATA_YEARS:
        area_pop_by_year[year] = [round(float(v), 0) for v in areas_proj[f'pop_{year}'].values]
        area_emp_by_year[year] = [round(float(v), 0) for v in areas_proj[f'emp_{year}'].values]
    # Default year for backwards compatibility
    area_pop = area_pop_by_year[DEFAULT_YEAR]
    area_emp = area_emp_by_year[DEFAULT_YEAR]

    # Education and transit data (not year-dependent)
    area_students = [round(float(v), 0) for v in areas_proj['students'].values]
    area_transit = [round(float(v), 0) for v in areas_proj['transit'].values]

    # Pre-compute area center nodes (which network node is closest to each area centroid)
    centroids_proj = [[round(c.x, 1), round(c.y, 1)] for c in areas_proj.geometry.centroid]
    area_center_nodes = []
    for c in areas_proj.geometry.centroid:
        _, idx = nt.query([c.x, c.y])
        area_center_nodes.append(ni[idx])

    version = get_version()
    print(f"Version: {version}")

    html = generate_html(
        areas_geojson=areas_geojson,
        completed_geojson=completed_geojson,
        construction_geojson=construction_geojson,
        plan_geojson=plan_geojson,
        check_geojson=check_geojson,
        wishing_geojson=wishing_geojson,
        lane_names=lane_names,
        lane_lengths=lane_lengths,
        area_names=area_names,
        area_pop=area_pop,
        area_emp=area_emp,
        area_pop_by_year=area_pop_by_year,
        area_emp_by_year=area_emp_by_year,
        area_students=area_students,
        area_transit=area_transit,
        data_years=DATA_YEARS,
        default_year=DEFAULT_YEAR,
        area_center_nodes=area_center_nodes,
        nodes_wgs=nodes_wgs,
        edges_list=edges_list,
        edge_geoms_wgs=edge_geoms_wgs,
        completed_edges=completed_edges,
        completed_virtual_edges=completed_virtual_edges,
        construction_edges=construction_edges,
        construction_virtual_edges=construction_virtual_edges,
        plan_edges=plan_edges,
        plan_virtual_edges=plan_virtual_edges,
        check_edges=check_edges,
        check_virtual_edges=check_virtual_edges,
        wishing_edges=wishing_edges,
        wishing_geoms=wishing_geoms,
        wishing_virtual_edges=wishing_virtual_edges,
        centroids_wgs=centroids_wgs,
        k_values=K_VALUES,
        theta_values=THETA_VALUES,
        version=version,
    )

    out_path = script_dir / 'bike_analysis.html'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Done! Open {out_path}")


def generate_html(*, areas_geojson, completed_geojson, construction_geojson,
                  plan_geojson, check_geojson,
                  wishing_geojson, lane_names, lane_lengths, area_names,
                  area_pop, area_emp, area_pop_by_year, area_emp_by_year,
                  area_students, area_transit,
                  data_years, default_year, area_center_nodes,
                  nodes_wgs, edges_list, edge_geoms_wgs,
                  completed_edges, completed_virtual_edges,
                  construction_edges, construction_virtual_edges,
                  plan_edges, plan_virtual_edges,
                  check_edges, check_virtual_edges,
                  wishing_edges, wishing_geoms,
                  wishing_virtual_edges, centroids_wgs,
                  k_values, theta_values, version='dev'):

    # Serialize data compactly
    def js_json(obj):
        return json.dumps(obj, ensure_ascii=False, separators=(',', ':'))

    # Generate K options (default 10)
    k_options = '\n'.join([
        f'      <option value="{k}"{" selected" if k == 10 else ""}>{k}</option>'
        for k in k_values
    ])
    k_options += '\n      <option value="custom">Custom...</option>'

    # Generate theta options (default -1.0)
    theta_options = '\n'.join([
        f'      <option value="{t}"{" selected" if t == -1.0 else ""}>{t}</option>'
        for t in theta_values
    ])
    theta_options += '\n      <option value="custom">Custom...</option>'

    # Generate year options
    year_options = '\n'.join([
        f'      <option value="{y}"{" selected" if y == default_year else ""}>{y}</option>'
        for y in data_years
    ])

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Jerusalem Bike Lane Analysis</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<link rel="stylesheet" href="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css"/>
<script src="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js"></script>
<style>
*{{box-sizing:border-box}}
body{{font-family:Arial,sans-serif;margin:0;display:flex;flex-direction:column;height:100vh}}
.header{{background:#2c3e50;color:#fff;padding:10px 20px;display:flex;justify-content:space-between;align-items:center}}
.header h1{{margin:0;font-size:1.4em}}
.controls{{background:#34495e;padding:10px 20px;display:flex;gap:20px;flex-wrap:wrap;align-items:center}}
.cg{{display:flex;align-items:center;gap:8px}}
.cg label{{color:#fff;font-weight:700;font-size:.9em}}
select,button{{padding:7px 12px;border:none;border-radius:4px;font-size:13px}}
button{{background:#3498db;color:#fff;cursor:pointer}}
button:hover{{background:#2980b9}}
.radio-group{{display:flex;gap:12px;align-items:center}}
.radio-group label{{color:#fff;font-weight:400;cursor:pointer;display:flex;align-items:center;gap:4px}}
.radio-group input{{cursor:pointer}}
.main{{display:flex;flex:1;overflow:hidden}}
.map-wrap{{flex:1;position:relative}}
#map{{width:100%;height:100%}}
.sidebar{{width:380px;background:#ecf0f1;overflow-y:auto;padding:12px;font-size:.9em;flex-shrink:0;transition:width .2s ease}}
.sidebar h3{{margin:0 0 8px;color:#2c3e50;border-bottom:2px solid #3498db;padding-bottom:4px}}
.tabs{{display:flex;gap:6px;margin-bottom:10px}}
.tabs button{{flex:1;padding:8px;text-align:center}}
.tabs button.act{{background:#27ae60}}
.tc{{display:none}}.tc.act{{display:block}}
.lane{{padding:6px 8px;margin:4px 0;background:#fff;border-radius:4px;cursor:pointer;display:flex;justify-content:space-between;align-items:center;border:2px solid transparent}}
.lane:hover{{background:#d5dbdb}}
.lane.sel{{background:#a9dfbf;border-color:#27ae60}}
.lane .pct{{font-weight:700;color:#27ae60;white-space:nowrap;margin-left:8px}}
.legend{{position:absolute;bottom:30px;right:10px;background:#fff;padding:10px;border-radius:5px;box-shadow:0 2px 5px rgba(0,0,0,.3);z-index:1000;font-size:.85em}}
.legend-item{{display:flex;align-items:center;gap:8px;margin:4px 0}}
.legend-line{{width:24px;height:4px;border-radius:2px}}
.info{{position:absolute;top:10px;left:10px;background:#fff;padding:10px 14px;border-radius:5px;box-shadow:0 2px 5px rgba(0,0,0,.3);z-index:1000;max-width:300px;display:none}}
.path-ctl{{background:#f8f9fa;padding:10px;border-radius:5px;margin-bottom:12px}}
.path-ctl select{{width:100%;margin:4px 0}}
.path-ctl button{{width:100%;margin-top:8px}}
.note{{background:#fff3cd;padding:8px;border-radius:4px;margin-bottom:10px;font-size:.85em}}
#totalImp{{font-size:.95em}}
.formula{{background:#1a252f;color:#ccc;padding:6px 20px;font-size:.85em;display:flex;align-items:center;gap:15px}}
.formula .math{{color:#fff;font-family:'Times New Roman',serif;font-size:1.1em}}
.formula .math .var{{color:#f1c40f}}
.path-stats{{background:#fff;border-radius:5px;padding:10px;margin-top:10px}}
.path-stats .bar{{height:20px;border-radius:3px;display:flex;overflow:hidden;margin:6px 0}}
.path-stats .bar-lane{{background:#1565C0}}
.path-stats .bar-road{{background:#E65100}}
.path-stats table{{width:100%;border-collapse:collapse;font-size:.9em}}
.path-stats td{{padding:3px 6px}}
.path-stats td:last-child{{text-align:right;font-weight:700}}
.user-lane{{padding:8px;margin:4px 0;background:#fff;border-radius:4px;border:2px solid #e91e63;display:flex;flex-direction:column;gap:4px}}
.user-lane.active{{background:#fce4ec;border-color:#c2185b}}
.user-lane .lane-header{{display:flex;justify-content:space-between;align-items:center}}
.user-lane .lane-name{{font-weight:700;color:#c2185b;flex:1}}
.user-lane .lane-length{{color:#666;font-size:.85em;margin-right:8px}}
.user-lane .lane-actions{{display:flex;gap:4px}}
.user-lane .lane-actions button{{padding:3px 8px;font-size:11px;background:#e91e63;color:#fff;border:none;border-radius:3px;cursor:pointer}}
.user-lane .lane-actions button:hover{{background:#c2185b}}
.user-lane .lane-actions button.del{{background:#e74c3c}}
.user-lane .lane-actions button.del:hover{{background:#c0392b}}
.draw-controls{{background:#fce4ec;padding:10px;border-radius:5px;margin-bottom:10px;border:1px solid #e91e63}}
.draw-controls button{{margin:4px 2px}}
.draw-note{{background:#f8bbd9;padding:8px;border-radius:4px;margin-bottom:10px;font-size:.85em;border-left:3px solid #e91e63}}
.impact-box{{background:#e8f5e9;padding:10px;border-radius:5px;margin-top:10px;border:1px solid #4caf50}}
.impact-box.negative{{background:#ffebee;border-color:#f44336}}
.impact-box h4{{margin:0 0 8px;color:#2e7d32}}
.impact-box.negative h4{{color:#c62828}}
.export-section{{margin-top:12px;padding-top:12px;border-top:1px solid #ddd}}
.sidebar-toggle{{display:none;position:fixed;bottom:20px;right:20px;z-index:2000;width:50px;height:50px;border-radius:50%;background:#3498db;color:#fff;border:none;font-size:24px;box-shadow:0 2px 10px rgba(0,0,0,.4);cursor:pointer}}
.panel-btns{{display:flex;gap:4px;flex-shrink:0}}
.hdr-btn{{padding:3px 8px;font-size:11px;background:rgba(255,255,255,0.15);border:1px solid rgba(255,255,255,0.3);border-radius:3px;color:#ccc;cursor:pointer}}
.hdr-btn:hover{{background:rgba(255,255,255,0.25);color:#fff}}
.formula.collapsed,.controls.collapsed{{display:none}}
.sidebar.collapsed{{width:0;min-width:0;padding:0;overflow:hidden}}
.resize-handle{{width:5px;background:#bdc3c7;cursor:col-resize;flex-shrink:0;transition:background .15s}}
.resize-handle:hover,.resize-handle.dragging{{background:#7f8c8d}}
.legend-header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;cursor:pointer}}
.legend-header strong{{flex:1}}
.legend-toggle{{background:none;border:none;padding:0 2px;font-size:14px;color:#666;cursor:pointer;line-height:1}}
#legendContent.collapsed{{display:none}}
@media(max-width:768px){{
  .header{{flex-direction:column;gap:4px;padding:8px 12px}}
  .header h1{{font-size:1.1em}}
  .formula{{display:none}}
  #btnFormula{{display:none!important}}
  .controls{{padding:8px 12px;gap:10px}}
  .cg label{{font-size:.8em}}
  .main{{flex-direction:column}}
  .map-wrap{{flex:1;min-height:200px}}
  .sidebar{{width:100%;height:45vh;overflow-y:auto;flex-shrink:0;transition:height .2s ease}}
  .resize-handle{{width:100%;height:5px;cursor:row-resize}}
  .sidebar.mobile-hidden{{display:none}}
  .sidebar-toggle{{display:block}}
  .method-btn-mobile{{display:inline-block!important}}
  .legend{{font-size:.75em;max-width:180px;bottom:70px}}
  .info{{max-width:200px;font-size:.85em}}
  .tabs button{{padding:6px 4px;font-size:.75em}}
  #totalImp{{font-size:.8em}}
}}
@media(max-width:768px){{
  .sidebar.collapsed{{width:100%;height:0;min-height:0;padding:0;overflow:hidden}}
}}

@media(max-width:480px){{
  .header h1{{font-size:.95em}}
  .controls{{flex-direction:column;gap:8px;align-items:flex-start}}
  .sidebar{{height:50vh}}
  .tabs{{flex-wrap:wrap}}
  .tabs button{{flex:none;font-size:.7em;padding:5px 6px}}
}}
</style>
</head>
<body>
<div class="header">
  <h1>Jerusalem Bike Lane Analysis <span style="font-size:12px;color:#7f8c8d;font-weight:normal">v{version}</span></h1>
  <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
    <span id="totalImp">Estimated total improvement: 0%</span>
    <div class="panel-btns">
      <button class="hdr-btn" id="btnFormula" onclick="togglePanelSection('formula','btnFormula','Formula')">Formula &#9660;</button>
      <button class="hdr-btn" id="btnParams" onclick="togglePanelSection('controls','btnParams','Params')">Params &#9660;</button>
      <button class="hdr-btn" id="btnSidebar" onclick="toggleSidebarDesktop()">Panel &#9660;</button>
    </div>
    <button class="method-btn-mobile" onclick="document.getElementById('methodModal').style.display='flex'" style="display:none;padding:4px 10px;background:#27ae60;color:#fff;border-radius:4px;border:none;font-size:11px;cursor:pointer">Methodology</button>
  </div>
</div>
<div class="formula">
  <span>Accessibility model:</span>
  <span class="math">
    N = &Sigma;<sub>i</sub> &Sigma;<sub>j</sub> P<sub>i</sub> &middot; D<sub>j</sub> &middot; &tau;<sub>ij</sub><sup class="var">&theta;</sup>
    &nbsp;&nbsp;where&nbsp;
    D<sub>j</sub> = w<sub>emp</sub>&middot;E + w<sub>edu</sub>&middot;S + w<sub>transit</sub>&middot;T,&nbsp;
    &tau;<sub>ij</sub> = shortest path with weight
    <span class="var">K</span>&middot;comfort&middot;d for roads,&nbsp; 1&middot;d for bike lanes
  </span>
  <button onclick="document.getElementById('methodModal').style.display='flex'" style="margin-left:20px;padding:5px 12px;background:#27ae60;color:#fff;border-radius:4px;border:none;font-size:12px;cursor:pointer">Methodology</button>
</div>
<div class="controls">
  <div class="cg">
    <label>K (no-lane penalty):</label>
    <select id="kSel" onchange="handleKChange()">
{k_options}
    </select>
    <input type="number" id="kCustom" placeholder="Enter K" min="1" max="10000" style="width:80px;display:none" onchange="applyCustomK()">
  </div>
  <div class="cg">
    <label>&theta; (distance decay):</label>
    <select id="tSel" onchange="handleThetaChange()">
{theta_options}
    </select>
    <input type="number" id="tCustom" placeholder="Enter &theta;" step="0.1" min="-10" max="0" style="width:80px;display:none" onchange="applyCustomTheta()">
  </div>
  <div class="cg">
    <label>Year:</label>
    <select id="yearSel" onchange="handleYearChange()">
{year_options}
    </select>
  </div>
  <div class="cg">
    <label>Destination weights:</label>
    <div style="display:flex;flex-direction:column;gap:2px;font-size:11px">
      <div style="display:flex;align-items:center;gap:4px">
        <span style="width:62px;color:#fff">Employment</span>
        <input type="range" id="wEmpSlider" min="0" max="100" value="33" style="width:80px" oninput="updateWeights()">
        <span id="wEmpVal" style="width:30px">33%</span>
      </div>
      <div style="display:flex;align-items:center;gap:4px">
        <span style="width:62px;color:#fff">Education</span>
        <input type="range" id="wEduSlider" min="0" max="100" value="33" style="width:80px" oninput="updateWeights()">
        <span id="wEduVal" style="width:30px">33%</span>
      </div>
      <div style="display:flex;align-items:center;gap:4px">
        <span style="width:62px;color:#fff">Transit</span>
        <input type="range" id="wTransitSlider" min="0" max="100" value="33" style="width:80px" oninput="updateWeights()">
        <span id="wTransitVal" style="width:30px">33%</span>
      </div>
    </div>
  </div>
  <div class="cg">
    <label>Color areas by:</label>
    <div class="radio-group">
      <label><input type="radio" name="accMode" value="origin" checked onchange="updateAreaColors()"> Origin</label>
      <label><input type="radio" name="accMode" value="dest" onchange="updateAreaColors()"> Destination</label>
    </div>
  </div>
  <div class="cg">
    <label>Show:</label>
    <div class="radio-group">
      <label><input type="radio" name="showMode" value="accessibility" checked onchange="updateAreaColors()"> Accessibility</label>
      <label><input type="radio" name="showMode" value="change" onchange="updateAreaColors()"> Change (%)</label>
    </div>
  </div>
</div>
<div class="main">
  <div class="map-wrap">
    <div id="map"></div>
    <div class="legend">
      <div class="legend-header" onclick="toggleLegend()">
        <strong>Legend</strong>
        <button class="legend-toggle" id="legendToggle">&#9660;</button>
      </div>
      <div id="legendContent">
        <div class="legend-item"><label style="display:flex;align-items:center;gap:6px;cursor:pointer"><input type="checkbox" id="lyrCompleted" checked onchange="toggleLayer('completed')"><div class="legend-line" style="background:#1B5E20"></div>Existing lanes</label></div>
        <div class="legend-item"><label style="display:flex;align-items:center;gap:6px;cursor:pointer"><input type="checkbox" id="lyrConstruction" checked onchange="toggleLayer('construction')"><div class="legend-line" style="background:#81C784"></div>Under construction</label></div>
        <div class="legend-item"><label style="display:flex;align-items:center;gap:6px;cursor:pointer"><input type="checkbox" id="lyrPlan" onchange="toggleLayer('plan')"><div class="legend-line" style="background:#2196F3"></div>In planning</label></div>
        <div class="legend-item"><label style="display:flex;align-items:center;gap:6px;cursor:pointer"><input type="checkbox" id="lyrCheck" onchange="toggleLayer('check')"><div class="legend-line" style="background:#00BCD4"></div>In checking</label></div>
        <div class="legend-item"><label style="display:flex;align-items:center;gap:6px;cursor:pointer"><input type="checkbox" id="lyrWishing" onchange="toggleLayer('wishing')"><div class="legend-line" style="background:#FF9800"></div>Wishing list</label></div>
        <hr style="margin:6px 0;border:none;border-top:1px solid #ccc">
        <div class="legend-item"><div class="legend-line" style="background:#9b59b6;height:6px"></div>Selected wishing lane</div>
        <div class="legend-item"><div class="legend-line" style="background:#E91E63;height:6px"></div>User-drawn lane</div>
        <div class="legend-item"><div class="legend-line" style="background:#1565C0;height:6px"></div>Path on bike lane</div>
        <div class="legend-item"><div class="legend-line" style="background:#E65100;height:6px"></div>Path on road</div>
        <hr style="margin:6px 0;border:none;border-top:1px solid #ccc">
        <strong>Area Accessibility (log scale)</strong>
        <div class="legend-item" style="flex-direction:column;align-items:flex-start;gap:2px">
          <div style="display:flex;align-items:center;gap:4px">
            <div style="width:80px;height:12px;background:linear-gradient(to right,#0000CD,#00CED1,#FFFF00,#FFA500,#DC143C);border-radius:2px"></div>
          </div>
          <div style="display:flex;justify-content:space-between;width:80px;font-size:0.75em">
            <span>Low</span><span>High</span>
          </div>
        </div>
      </div>
    </div>
    <div class="info" id="info">
      <strong id="infoTitle"></strong>
      <div id="infoBody"></div>
    </div>
  </div>
  <div class="resize-handle" id="resizeHandle"></div>
  <div class="sidebar">
    <div class="tabs">
      <button class="act" onclick="showTab('lanes',this)">Select Lanes</button>
      <button onclick="showTab('draw',this)">Draw Lane</button>
      <button onclick="showTab('paths',this)">Find Path</button>
      <button onclick="showTab('compute',this)">Compute Accessibility</button>
      <button onclick="showTab('rank',this)">Rank Lanes</button>
    </div>
    <div id="lanes" class="tc act">
      <h3>Select Wishing Lanes</h3>
      <div class="note">Click lanes to select them for the network. Selected lanes affect path finding and accessibility calculations.</div>
      <input type="text" id="laneSearch" placeholder="Search lanes..." style="width:100%;padding:8px;margin:8px 0;border:1px solid #ddd;border-radius:4px;box-sizing:border-box" oninput="filterLanes()">
      <div style="margin:8px 0;display:flex;gap:8px">
        <button onclick="selectAllLanes()">Select All</button>
        <button onclick="clearSel()">Clear</button>
      </div>
      <div id="laneList"></div>
    </div>
    <div id="draw" class="tc">
      <h3>Draw Custom Lanes</h3>
      <div class="draw-note">Draw your own bike lanes on the map to test their impact on accessibility. Click points to create a lane path, double-click to finish.</div>
      <div class="draw-controls">
        <button onclick="startDrawing()" id="drawBtn">Start Drawing</button>
        <button onclick="cancelDrawing()" id="cancelBtn" style="display:none;background:#e74c3c">Cancel</button>
        <button onclick="finishDrawing()" id="finishBtn" style="display:none;background:#27ae60">Finish Lane</button>
      </div>
      <div id="drawingStatus" style="margin:8px 0;font-size:.85em;color:#666"></div>
      <h4 style="margin:12px 0 8px;color:#c2185b">Your Drawn Lanes (<span id="userLaneCount">0</span>)</h4>
      <div id="userLaneList"></div>
      <div id="userLaneImpact"></div>
      <div class="export-section">
        <button onclick="exportUserLanes()" style="width:48%">Export GeoJSON</button>
        <button onclick="document.getElementById('importFile').click()" style="width:48%">Import GeoJSON</button>
        <input type="file" id="importFile" accept=".geojson,.json" style="display:none" onchange="importUserLanes(event)">
      </div>
    </div>
    <div id="compute" class="tc">
      <h3>Compute Accessibility</h3>
      <div class="note">Calculate accessibility for current network with selected lanes. This computes all area-to-area distances (30-60 sec).</div>
      <div class="path-ctl">
        <p><b>Selected lanes:</b> <span id="selCount">0</span></p>
        <p><b>Parameters:</b> K=<span id="compK">100</span>, &theta;=<span id="compT">-1.0</span>, Year=<span id="compYear">2025</span></p>
        <button onclick="computeAccessibility()" id="computeBtn">Compute Accessibility</button>
        <div id="computeProgress" style="margin-top:10px"></div>
      </div>
      <div id="computeResults" style="margin-top:10px"></div>
    </div>
    <div id="paths" class="tc">
      <h3>Find Shortest Path</h3>
      <div class="note">Compute shortest path using current network (with selected lanes).</div>
      <div style="margin-bottom:10px">
        <label style="margin-right:15px"><input type="radio" name="pathMode" value="area" checked onchange="togglePathMode()"> By area</label>
        <label><input type="radio" name="pathMode" value="point" onchange="togglePathMode()"> By map point</label>
      </div>
      <div id="areaPathMode" class="path-ctl">
        <label>Origin area:</label>
        <input type="text" id="origInput" list="areaList" placeholder="Search area..." style="width:100%;padding:6px;margin-bottom:8px;border:1px solid #ddd;border-radius:4px">
        <label>Destination area:</label>
        <input type="text" id="destInput" list="areaList" placeholder="Search area..." style="width:100%;padding:6px;margin-bottom:8px;border:1px solid #ddd;border-radius:4px">
        <datalist id="areaList"></datalist>
        <button onclick="showPath()">Show path</button>
      </div>
      <div id="pointPathMode" style="display:none" class="path-ctl">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
          <label style="min-width:70px">Origin:</label>
          <span id="origPointLabel" style="flex:1;padding:6px;background:#f5f5f5;border:1px solid #ddd;border-radius:4px;font-size:.85em;color:#666">Not set - click Pick</span>
          <button onclick="startPickingPoint('origin')" style="padding:4px 10px">Pick</button>
          <button onclick="clearPathPoint('origin')" style="padding:4px 8px;background:#f5f5f5">X</button>
        </div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
          <label style="min-width:70px">Dest:</label>
          <span id="destPointLabel" style="flex:1;padding:6px;background:#f5f5f5;border:1px solid #ddd;border-radius:4px;font-size:.85em;color:#666">Not set - click Pick</span>
          <button onclick="startPickingPoint('dest')" style="padding:4px 10px">Pick</button>
          <button onclick="clearPathPoint('dest')" style="padding:4px 8px;background:#f5f5f5">X</button>
        </div>
        <div id="pickingStatus" style="font-size:.85em;color:#e91e63;margin-bottom:8px"></div>
        <button onclick="showPath()">Show path</button>
      </div>
      <div id="pathInfo"></div>
    </div>
    <div id="rank" class="tc">
      <h3>Rank Lanes</h3>
      <div class="note">Calculate the contribution of each lane to network accessibility. Choose a ranking mode below.</div>
      <div style="margin:10px 0">
        <label style="display:block;margin:6px 0;cursor:pointer">
          <input type="radio" name="rankMode" value="additive" checked onchange="updateRankModeDescription()">
          <b>Additive</b> - Compare current network to adding each lane
        </label>
        <label style="display:block;margin:6px 0;cursor:pointer">
          <input type="radio" name="rankMode" value="subtractive" onchange="updateRankModeDescription()">
          <b>Subtractive</b> - Compare current network to removing each lane
        </label>
      </div>
      <div id="rankModeDescription" class="note" style="background:#fff3cd;padding:8px;border-radius:4px;margin:10px 0">
        <b>Additive mode:</b> Measures the value each lane would add if built next. Baseline is the currently selected lanes.
      </div>
      <div class="path-ctl">
        <p><b>Parameters:</b> K=<span id="rankK">100</span>, &theta;=<span id="rankT">-1.0</span>, Year=<span id="rankYear">2025</span></p>
        <button onclick="computeRankLanes()" id="rankBtn">Compute Rankings</button>
        <div id="rankProgress" style="margin-top:10px"></div>
      </div>
      <div id="rankResults" style="margin-top:10px"></div>
    </div>
  </div>
</div>

<script>
// === EMBEDDED DATA ===
const AREAS={js_json(areas_geojson)};
const COMPLETED={js_json(completed_geojson)};
const CONSTRUCTION={js_json(construction_geojson)};
const PLAN={js_json(plan_geojson)};
const CHECK={js_json(check_geojson)};
const WISHING={js_json(wishing_geojson)};
const LANE_NAMES={js_json(lane_names)};
const LANE_LENGTHS={js_json(lane_lengths)};
const AREA_NAMES={js_json(area_names)};
let AREA_POP={js_json(area_pop)};
let AREA_EMP={js_json(area_emp)};
const AREA_POP_BY_YEAR={js_json(area_pop_by_year)};
const AREA_EMP_BY_YEAR={js_json(area_emp_by_year)};
const AREA_STUDENTS={js_json(area_students)};
const AREA_TRANSIT={js_json(area_transit)};
const DATA_YEARS={js_json(data_years)};
const DEFAULT_YEAR={default_year};
const AREA_NODES={js_json(area_center_nodes)};
const NODES={js_json(nodes_wgs)};
const EDGES={js_json(edges_list)};
const EDGE_GEOMS={js_json(edge_geoms_wgs)};
// Per-layer edge sets: which road edges each layer covers
const LAYER_EDGES={{
  completed:{js_json(completed_edges)},
  construction:{js_json(construction_edges)},
  plan:{js_json(plan_edges)},
  check:{js_json(check_edges)}
}};
// Per-layer virtual edges
const LAYER_VIRTUAL_EDGES={{
  completed:{js_json(completed_virtual_edges)},
  construction:{js_json(construction_virtual_edges)},
  plan:{js_json(plan_virtual_edges)},
  check:{js_json(check_virtual_edges)}
}};
const WISHING_EDGES={js_json(wishing_edges)};
const WISHING_GEOMS={js_json(wishing_geoms)};
const WISHING_VIRTUAL_EDGES={js_json(wishing_virtual_edges)};
const CENTROIDS={js_json(centroids_wgs)};

// All computation is done online - no pre-computed accessibility data

// === STATE ===
const sel=new Set();
let wishLyr,areasLyr,pathLyrGroup;
let completedLyr,constructionLyr,planLyr,checkLyr;
// Layer toggle state: which layers are included in the network for calculations
const activeLayers={{completed:true,construction:true,plan:false,check:false,wishing:false}};
let currentK=10;
let currentTheta=-1.0;
let currentYear=DEFAULT_YEAR;

// Destination layer weights (normalized to sum to 1)
let wEmp=1/3,wEdu=1/3,wTransit=1/3;
function destWeight(j){{return wEmp*AREA_EMP[j]+wEdu*AREA_STUDENTS[j]+wTransit*AREA_TRANSIT[j];}}

// Baseline = accessibility with NO wishing lanes (computed when K/theta/year changes)
let baselineAcc=null;
let baselineK=null;
let baselineTheta=null;
let baselineYear=null;
let baselineWEmp=null;
let baselineWEdu=null;
let baselineWTransit=null;

// Computed = accessibility WITH selected wishing lanes
let computedAcc=null;
let computedK=null;
let computedTheta=null;
let computedYear=null;
let computedSel=null;

// Path point selection state
let pathOriginPoint=null;
let pathDestPoint=null;
let pathOriginMarker=null;
let pathDestMarker=null;
let pickingPointFor=null;

// User-drawn lanes
const userLanes=[];  // Array of {{id, name, coords, length, active, layer, edges}}
let userLaneIdCounter=0;
let userLanesLyrGroup=null;
let drawControl=null;
let currentDrawLayer=null;
let isDrawing=false;

// === CUSTOM PARAMETER HANDLERS ===
function getK(){{return currentK;}}
function getTheta(){{return currentTheta;}}

function handleKChange(){{
  const s=document.getElementById("kSel");
  const custom=document.getElementById("kCustom");
  if(s.value==="custom"){{
    custom.style.display="inline";
    custom.focus();
  }}else{{
    custom.style.display="none";
    currentK=parseFloat(s.value);
    onParamsChanged();
  }}
}}

function applyCustomK(){{
  const val=parseFloat(document.getElementById("kCustom").value);
  if(!isNaN(val)&&val>0){{
    currentK=val;
    onParamsChanged();
  }}
}}

function handleThetaChange(){{
  const s=document.getElementById("tSel");
  const custom=document.getElementById("tCustom");
  if(s.value==="custom"){{
    custom.style.display="inline";
    custom.focus();
  }}else{{
    custom.style.display="none";
    currentTheta=parseFloat(s.value);
    onParamsChanged();
  }}
}}

function applyCustomTheta(){{
  const val=parseFloat(document.getElementById("tCustom").value);
  if(!isNaN(val)&&val<0){{
    currentTheta=val;
    onParamsChanged();
  }}
}}

function handleYearChange(){{
  const s=document.getElementById("yearSel");
  currentYear=parseInt(s.value);
  AREA_POP=AREA_POP_BY_YEAR[currentYear];
  AREA_EMP=AREA_EMP_BY_YEAR[currentYear];
  onParamsChanged();
}}

function updateWeights(){{
  const e=parseInt(document.getElementById('wEmpSlider').value);
  const d=parseInt(document.getElementById('wEduSlider').value);
  const t=parseInt(document.getElementById('wTransitSlider').value);
  const total=e+d+t;
  if(total===0){{wEmp=1;wEdu=0;wTransit=0;}}
  else{{wEmp=e/total;wEdu=d/total;wTransit=t/total;}}
  document.getElementById('wEmpVal').textContent=Math.round(wEmp*100)+'%';
  document.getElementById('wEduVal').textContent=Math.round(wEdu*100)+'%';
  document.getElementById('wTransitVal').textContent=Math.round(wTransit*100)+'%';
  onParamsChanged();
}}

function onParamsChanged(){{
  // Clear computed results when params change
  computedAcc=null;
  computedK=null;
  computedTheta=null;
  computedYear=null;
  computedSel=null;
  // Also clear baseline - will be recomputed when "Compute Accessibility" is clicked
  baselineAcc=null;
  baselineK=null;
  baselineTheta=null;
  baselineYear=null;
  baselineWEmp=null;
  baselineWEdu=null;
  baselineWTransit=null;
  refresh();
  updateComputePanel();
}}

function selectAllLanes(){{
  for(let i=0;i<LANE_NAMES.length;i++)sel.add(i);
  refresh();
}}

// === LAYER TOGGLE ===
const layerRefs={{completed:null,construction:null,plan:null,check:null}};
function toggleLayer(name){{
  const cb=document.getElementById('lyr'+name.charAt(0).toUpperCase()+name.slice(1));
  activeLayers[name]=cb.checked;
  // Show/hide map layer
  const lyrMap={{completed:completedLyr,construction:constructionLyr,plan:planLyr,check:checkLyr}};
  const lyr=lyrMap[name];
  if(lyr){{
    if(cb.checked){{lyr.addTo(map);}}
    else{{map.removeLayer(lyr);}}
  }}
  // Wishing layer: toggle visibility + include all wishing lanes in network when checked
  if(name==='wishing'){{
    if(cb.checked){{
      wishLyr.addTo(map);
      // Select all wishing lanes
      for(let i=0;i<LANE_NAMES.length;i++)sel.add(i);
    }}else{{
      map.removeLayer(wishLyr);
      sel.clear();
    }}
  }}
  onParamsChanged();
}}

// Build set of all edges covered by active layers (non-wishing)
function getActiveLayerEdgeSet(){{
  const edgeSet=new Set();
  for(const name of ['completed','construction','plan','check']){{
    if(!activeLayers[name])continue;
    const edges=LAYER_EDGES[name]||[];
    for(const e of edges){{
      const a=Math.min(e[0],e[1]),b=Math.max(e[0],e[1]);
      edgeSet.add(a+'_'+b);
    }}
  }}
  return edgeSet;
}}

// Build array of all virtual edges from active layers (non-wishing)
function getActiveLayerVirtualEdges(){{
  const result=[];
  for(const name of ['completed','construction','plan','check']){{
    if(!activeLayers[name])continue;
    const ves=LAYER_VIRTUAL_EDGES[name]||[];
    for(const ve of ves)result.push(ve);
  }}
  return result;
}}

// Build set of all edges covered by selected wishing lanes
function getWishingEdgeSet(){{
  const edgeSet=new Set();
  for(const lid of sel){{
    const edges=WISHING_EDGES[lid]||[];
    for(const e of edges){{
      const a=Math.min(e[0],e[1]),b=Math.max(e[0],e[1]);
      edgeSet.add(a+'_'+b);
    }}
  }}
  return edgeSet;
}}

// Build array of all virtual edges from selected wishing lanes
function getWishingVirtualEdges(){{
  const result=[];
  for(const lid of sel){{
    const ves=WISHING_VIRTUAL_EDGES[lid]||[];
    for(const ve of ves)result.push(ve);
  }}
  return result;
}}

// Get combined edge set of all active layers + selected wishing lanes
function getAllBikeEdgeSet(){{
  const s=getActiveLayerEdgeSet();
  const w=getWishingEdgeSet();
  for(const e of w)s.add(e);
  return s;
}}

// Get combined virtual edges of all active layers + selected wishing lanes
function getAllBikeVirtualEdges(){{
  return [...getActiveLayerVirtualEdges(),...getWishingVirtualEdges()];
}}

// Build adjacency list with given bike edge set and virtual edges
function buildAdj(bikeEdgeSet,virtualEdges,k,includeEdgeInfo){{
  const adj={{}};
  for(const e of EDGES){{
    const len=e[2];
    const a=String(e[0]),b=String(e[1]);
    const edgeKey=Math.min(e[0],e[1])+'_'+Math.max(e[0],e[1]);
    const bike=bikeEdgeSet.has(edgeKey);
    const discomfort=e[4]||1.0;
    const w=bike?len:len*k*discomfort;
    if(!adj[a])adj[a]=[];
    if(!adj[b])adj[b]=[];
    if(includeEdgeInfo){{
      adj[a].push({{n:b,w:w,len:len,bike:bike,eKey:edgeKey}});
      adj[b].push({{n:a,w:w,len:len,bike:bike,eKey:edgeKey}});
    }}else{{
      adj[a].push({{n:b,w:w}});
      adj[b].push({{n:a,w:w}});
    }}
  }}
  // Add virtual edges (always bike lanes)
  for(const ve of virtualEdges){{
    const a=String(ve.from||ve['from']),b=String(ve.to||ve['to']);
    const len=ve.len;
    if(!adj[a])adj[a]=[];
    if(!adj[b])adj[b]=[];
    if(includeEdgeInfo){{
      const eKey='ve_'+a+'_'+b;
      adj[a].push({{n:b,w:len,len:len,bike:true,eKey:eKey,isVirtual:true}});
      adj[b].push({{n:a,w:len,len:len,bike:true,eKey:eKey,isVirtual:true}});
    }}else{{
      adj[a].push({{n:b,w:len}});
      adj[b].push({{n:a,w:len}});
    }}
  }}
  return adj;
}}

// === MAP INIT ===
const map=L.map("map").setView([31.78,35.22],12);
const baseCarto=L.tileLayer("https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png",{{
  attribution:'&copy; <a href="https://carto.com/">CARTO</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
  subdomains:'abcd',
  maxZoom:20
}}).addTo(map);
const baseSatellite=L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}",{{
  attribution:'&copy; Esri, Maxar, Earthstar Geographics',
  maxZoom:19
}});
L.control.layers({{"Map":baseCarto,"Satellite":baseSatellite}},null,{{position:'topright'}}).addTo(map);

// Areas
areasLyr=L.geoJSON(AREAS,{{
  style:()=>({{fillColor:"#3498db",weight:1,opacity:.5,color:"#2c3e50",fillOpacity:.1}}),
  onEachFeature:(f,layer)=>{{
    const p=f.properties;
    const aid=p.area_id;
    layer.on('click',function(e){{
      // Don't show popup when drawing a lane
      if(isDrawing)return;
      // If picking path points, forward the click to the point picker
      if(pickingPointFor){{onPathPointClick(e);return;}}
      const name=AREA_NAMES[aid]||"Area "+aid;
      const mode=getAccMode();
      let html="<b>"+name+"</b><br>"+
        "Pop: "+Math.round(AREA_POP[aid]).toLocaleString()+"<br>"+
        "Emp: "+Math.round(AREA_EMP[aid]).toLocaleString()+
        (AREA_STUDENTS[aid]>0?"<br>Students: "+Math.round(AREA_STUDENTS[aid]).toLocaleString():"")+
        (AREA_TRANSIT[aid]>0?"<br>Transit: "+Math.round(AREA_TRANSIT[aid]).toLocaleString()+" daily":"");
      html+="<hr style='margin:4px 0'>";
      // Show baseline
      if(baselineAcc && baselineK===currentK && baselineTheta===currentTheta && baselineYear===currentYear && baselineWEmp===wEmp && baselineWEdu===wEdu && baselineWTransit===wTransit){{
        const baseAcc=(mode==="dest")?baselineAcc.dest:baselineAcc.orig;
        html+="Baseline: "+baseAcc[aid].toFixed(1)+"<br>";
        // Show computed and change if available
        if(computedAcc && computedK===currentK && computedTheta===currentTheta && computedYear===currentYear){{
          const compAcc=(mode==="dest")?computedAcc.dest:computedAcc.orig;
          html+="With lanes: "+compAcc[aid].toFixed(1)+"<br>";
          if(baseAcc[aid]>0){{
            const changePct=100*(compAcc[aid]-baseAcc[aid])/baseAcc[aid];
            html+="Change: <span style='color:"+(changePct>=0?"#27ae60":"#e74c3c")+";font-weight:bold'>"+(changePct>=0?"+":"")+changePct.toFixed(2)+"%</span>";
          }}
        }}else{{
          html+="<i>Select lanes & compute to see change</i>";
        }}
      }}else{{
        html+="<i>Select lanes & compute to see results</i>";
      }}
      L.popup().setLatLng(e.latlng).setContent(html).openOn(map);
    }});
  }}
}}).addTo(map);

// Helper to create a lane layer with popup
function makeLaneLayer(data,color,label){{
  return L.geoJSON(data,{{style:{{color:color,weight:3,opacity:.8}},
    onEachFeature:(f,l)=>{{
      const content="<b>"+label+":</b> "+(f.properties.Name||"");
      l.on('click',function(e){{
        if(isDrawing)return;
        if(pickingPointFor){{onPathPointClick(e);return;}}
        L.popup().setLatLng(e.latlng).setContent(content).openOn(map);
      }});
    }}
  }});
}}

// Completed (dark green) - on by default
if(COMPLETED.features.length){{
  completedLyr=makeLaneLayer(COMPLETED,"#1B5E20","Existing");
  completedLyr.addTo(map);
}}

// Construction (light green) - on by default
if(CONSTRUCTION.features.length){{
  constructionLyr=makeLaneLayer(CONSTRUCTION,"#81C784","Under construction");
  constructionLyr.addTo(map);
}}

// Plan (blue) - off by default
if(PLAN.features.length){{
  planLyr=makeLaneLayer(PLAN,"#2196F3","In planning");
}}

// Check (cyan) - off by default
if(CHECK.features.length){{
  checkLyr=makeLaneLayer(CHECK,"#00BCD4","In checking");
}}

// Wishing list (orange, purple when selected)
wishLyr=L.geoJSON(WISHING,{{
  style:f=>{{
    const s=sel.has(f.properties.lane_id);
    return {{color:s?"#9b59b6":"#FF9800",weight:s?5:3,opacity:.8}};
  }},
  onEachFeature:(f,layer)=>{{
    const lid=f.properties.lane_id;
    layer.on("click",(e)=>{{
      // Don't toggle selection when drawing a lane
      if(isDrawing)return;
      // If picking path points, forward the click to the point picker
      if(pickingPointFor){{onPathPointClick(e);return;}}
      sel.has(lid)?sel.delete(lid):sel.add(lid);refresh();
    }});
    layer.on("mouseover",()=>showInfo(lid));
    layer.on("mouseout",hideInfo);
  }}
}}).addTo(map);

// Populate area datalist (sorted alphabetically)
const areaNameToId={{}};
(function(){{
  const items=AREA_NAMES.map((n,i)=>({{id:i,name:n}}));
  items.sort((a,b)=>a.name.localeCompare(b.name,'he'));
  items.forEach(it=>areaNameToId[it.name]=it.id);
  const opts=items.map(it=>'<option value="'+it.name+'">').join("");
  document.getElementById("areaList").innerHTML=opts;
}})();

// === REFRESH ===
function refresh(){{
  // Update wishing layer style
  wishLyr.setStyle(f=>{{
    const s=sel.has(f.properties.lane_id);
    return {{color:s?"#9b59b6":"#FF9800",weight:s?5:3,opacity:.8}};
  }});
  // Bring selected lanes to front so they're visible above other layers
  wishLyr.eachLayer(function(layer){{
    if(layer.feature && sel.has(layer.feature.properties.lane_id)){{
      layer.bringToFront();
    }}
  }});
  // Update lane list
  buildLaneList();
  // Update total improvement display
  const activeCount=['completed','construction','plan','check'].filter(n=>activeLayers[n]).length;
  document.getElementById("totalImp").textContent="Active layers: "+activeCount+"/4 | Wishing lanes: "+sel.size+" of "+LANE_NAMES.length;
  // Update area colors
  updateAreaColors();
  // Update compute panel
  updateComputePanel();
}}

function buildLaneList(){{
  const search=(document.getElementById("laneSearch").value||"").toLowerCase();
  // Build alphabetically sorted list
  const items=LANE_NAMES.map((name,i)=>({{id:i,name:name}}));
  items.sort((a,b)=>a.name.localeCompare(b.name,'he'));
  // Filter by search
  const filtered=search?items.filter(it=>it.name.toLowerCase().includes(search)):items;
  const container=document.getElementById("laneList");
  container.innerHTML=filtered.map(it=>{{
    const cls=sel.has(it.id)?"lane sel":"lane";
    return '<div class="'+cls+'" onclick="toggleLane('+it.id+')"><span>'+it.name+'</span></div>';
  }}).join("");
}}

function filterLanes(){{
  buildLaneList();
}}

function toggleLane(id){{
  sel.has(id)?sel.delete(id):sel.add(id);
  refresh();
}}

function clearSel(){{sel.clear();refresh();updateComputePanel();}}

function getAccMode(){{
  const radio=document.querySelector('input[name="accMode"]:checked');
  return radio?radio.value:"origin";
}}

function updateAreaColors(){{
  const mode=getAccMode();
  const showMode=document.querySelector('input[name="showMode"]:checked')?.value||"accessibility";

  // If showing "change" mode, need both baseline and computed
  if(showMode==="change"){{
    if(!baselineAcc || !computedAcc || baselineK!==currentK || baselineTheta!==currentTheta ||
       computedK!==currentK || computedTheta!==currentTheta){{
      // Need to compute first
      areasLyr.eachLayer(layer=>{{
        layer.setStyle({{fillColor:"#3498db",fillOpacity:.1,weight:1,opacity:.5,color:"#2c3e50"}});
      }});
      return;
    }}
    // Show percentage change from baseline
    const base=(mode==="dest")?baselineAcc.dest:baselineAcc.orig;
    const comp=(mode==="dest")?computedAcc.dest:computedAcc.orig;
    if(!base||!comp)return;

    // Calculate percentage changes
    const deltaPct=comp.map((v,i)=>base[i]>0?100*(v-base[i])/base[i]:0);
    const posDeltas=deltaPct.filter(d=>d>0);
    if(posDeltas.length===0){{
      // No positive changes
      areasLyr.eachLayer(layer=>{{
        layer.setStyle({{fillColor:"#BEBEBE",fillOpacity:.3,weight:1,opacity:.5,color:"#2c3e50"}});
      }});
      return;
    }}
    // Use log scale for positive changes
    const logDeltas=posDeltas.map(d=>Math.log(d+1));
    const maxLog=Math.max(...logDeltas);
    areasLyr.eachLayer(layer=>{{
      const aid=layer.feature.properties.area_id;
      const v=Math.max(0,deltaPct[aid]||0);
      if(v<=0){{
        layer.setStyle({{fillColor:"#BEBEBE",fillOpacity:.3,weight:1,opacity:.5,color:"#2c3e50"}});
        return;
      }}
      const logV=Math.log(v+1);
      const n=maxLog>0?logV/maxLog:0;
      const color=spectralColor(n);
      layer.setStyle({{fillColor:color,fillOpacity:.6,weight:1,opacity:.5,color:"#2c3e50"}});
    }});
    return;
  }}

  // Accessibility mode: show baseline if no computed, or computed if available
  let acc=null;
  if(computedAcc && computedK===currentK && computedTheta===currentTheta && computedYear===currentYear){{
    acc=(mode==="dest")?computedAcc.dest:computedAcc.orig;
  }}else if(baselineAcc && baselineK===currentK && baselineTheta===currentTheta && baselineYear===currentYear && baselineWEmp===wEmp && baselineWEdu===wEdu && baselineWTransit===wTransit){{
    acc=(mode==="dest")?baselineAcc.dest:baselineAcc.orig;
  }}

  if(!acc||!acc.length){{
    // No data - neutral coloring
    areasLyr.eachLayer(layer=>{{
      layer.setStyle({{fillColor:"#3498db",fillOpacity:.1,weight:1,opacity:.5,color:"#2c3e50"}});
    }});
    return;
  }}

  const pos=acc.filter(a=>a>0);
  if(!pos.length)return;
  // Use natural log for scaling
  const logVals=pos.map(v=>Math.log(v));
  const mnLog=Math.min(...logVals),mxLog=Math.max(...logVals);
  areasLyr.eachLayer(layer=>{{
    const aid=layer.feature.properties.area_id;
    const v=acc[aid]||0;
    if(v<=0){{
      layer.setStyle({{fillColor:"#BEBEBE",fillOpacity:.5,weight:1,opacity:.5,color:"#2c3e50"}});
      return;
    }}
    const logV=Math.log(v);
    const n=mxLog>mnLog?(logV-mnLog)/(mxLog-mnLog):0;
    const color=spectralColor(n);
    layer.setStyle({{fillColor:color,fillOpacity:.5,weight:1,opacity:.5,color:"#2c3e50"}});
  }});
}}

// Spectral colormap function matching reference image
function spectralColor(t){{
  // t goes from 0 (low) to 1 (high)
  // Colors: blue -> cyan -> yellow -> orange -> red (no green)
  const stops=[
    [0.0, 0,0,205],      // #0000CD dark blue
    [0.25, 0,206,209],   // #00CED1 cyan
    [0.5, 255,255,0],    // #FFFF00 yellow
    [0.75, 255,165,0],   // #FFA500 orange
    [1.0, 220,20,60]     // #DC143C red
  ];
  // Find segment
  let i=0;
  while(i<stops.length-1 && stops[i+1][0]<t)i++;
  if(i>=stops.length-1)return "rgb("+stops[stops.length-1][1]+","+stops[stops.length-1][2]+","+stops[stops.length-1][3]+")";
  const t0=stops[i][0],t1=stops[i+1][0];
  const f=(t-t0)/(t1-t0);
  const r=Math.round(stops[i][1]+(stops[i+1][1]-stops[i][1])*f);
  const g=Math.round(stops[i][2]+(stops[i+1][2]-stops[i][2])*f);
  const b=Math.round(stops[i][3]+(stops[i+1][3]-stops[i][3])*f);
  return "rgb("+r+","+g+","+b+")";
}}

function showInfo(lid){{
  const panel=document.getElementById("info");
  document.getElementById("infoTitle").textContent=LANE_NAMES[lid];
  document.getElementById("infoBody").textContent=sel.has(lid)?"Selected":"Click to select";
  panel.style.display="block";
}}
function hideInfo(){{document.getElementById("info").style.display="none";}}

// === TABS ===
function showTab(id,btn){{
  document.querySelectorAll(".tc").forEach(e=>e.classList.remove("act"));
  document.querySelectorAll(".tabs button").forEach(b=>b.classList.remove("act"));
  document.getElementById(id).classList.add("act");
  btn.classList.add("act");
}}

// === PATH FINDING ===
function togglePathMode(){{
  const mode=document.querySelector('input[name="pathMode"]:checked').value;
  document.getElementById('areaPathMode').style.display=mode==='area'?'block':'none';
  document.getElementById('pointPathMode').style.display=mode==='point'?'block':'none';
  if(pickingPointFor){{
    map.off('click',onPathPointClick);
    map.getContainer().style.cursor='';
    pickingPointFor=null;
    document.getElementById('pickingStatus').innerHTML='';
  }}
}}

function startPickingPoint(which){{
  if(pickingPointFor){{
    map.off('click',onPathPointClick);
  }}
  pickingPointFor=which;
  map.getContainer().style.cursor='crosshair';
  document.getElementById('pickingStatus').innerHTML='<b>Click on the map to set '+(which==='origin'?'origin':'destination')+' point...</b>';
  map.on('click',onPathPointClick);
}}

function onPathPointClick(e){{
  if(!pickingPointFor)return;

  const lat=e.latlng.lat,lng=e.latlng.lng;

  // Find nearest network node
  let nearestNode=null,nearestDist=Infinity;
  for(const[nid,c]of Object.entries(NODES)){{
    const d=haversineDistance(lat,lng,c[1],c[0]);
    if(d<nearestDist){{
      nearestDist=d;
      nearestNode=nid;
    }}
  }}

  if(!nearestNode){{
    alert('Could not find a nearby network node.');
    return;
  }}

  const nodeCoord=NODES[nearestNode];
  const point={{lat:lat,lng:lng,nodeId:nearestNode,nodeLat:nodeCoord[1],nodeLon:nodeCoord[0],dist:nearestDist}};

  if(pickingPointFor==='origin'){{
    pathOriginPoint=point;
    if(pathOriginMarker)map.removeLayer(pathOriginMarker);
    pathOriginMarker=L.marker([lat,lng],{{
      icon:L.divIcon({{className:'',html:'<div style="background:#27ae60;width:14px;height:14px;border-radius:50%;border:2px solid white;box-shadow:0 1px 3px rgba(0,0,0,.4)"></div>',iconSize:[14,14],iconAnchor:[7,7]}})
    }}).addTo(map);
    pathOriginMarker.bindPopup('Origin: connects to node '+nearestNode+' ('+(nearestDist).toFixed(0)+'m away)');
    document.getElementById('origPointLabel').innerHTML='<span style="color:#27ae60">Set</span> (connects '+nearestDist.toFixed(0)+'m to network)';
  }}else{{
    pathDestPoint=point;
    if(pathDestMarker)map.removeLayer(pathDestMarker);
    pathDestMarker=L.marker([lat,lng],{{
      icon:L.divIcon({{className:'',html:'<div style="background:#e74c3c;width:14px;height:14px;border-radius:50%;border:2px solid white;box-shadow:0 1px 3px rgba(0,0,0,.4)"></div>',iconSize:[14,14],iconAnchor:[7,7]}})
    }}).addTo(map);
    pathDestMarker.bindPopup('Destination: connects to node '+nearestNode+' ('+(nearestDist).toFixed(0)+'m away)');
    document.getElementById('destPointLabel').innerHTML='<span style="color:#e74c3c">Set</span> (connects '+nearestDist.toFixed(0)+'m to network)';
  }}

  map.off('click',onPathPointClick);
  map.getContainer().style.cursor='';
  document.getElementById('pickingStatus').innerHTML='';
  pickingPointFor=null;
}}

function clearPathPoint(which){{
  if(which==='origin'){{
    pathOriginPoint=null;
    if(pathOriginMarker){{map.removeLayer(pathOriginMarker);pathOriginMarker=null;}}
    document.getElementById('origPointLabel').innerHTML='Not set - click Pick';
  }}else{{
    pathDestPoint=null;
    if(pathDestMarker){{map.removeLayer(pathDestMarker);pathDestMarker=null;}}
    document.getElementById('destPointLabel').innerHTML='Not set - click Pick';
  }}
}}

function showPath(){{
  const mode=document.querySelector('input[name="pathMode"]:checked').value;

  let oNode,dNode,origIdx,destIdx;

  if(mode==='area'){{
    const origName=document.getElementById("origInput").value;
    const destName=document.getElementById("destInput").value;
    origIdx=areaNameToId[origName];
    destIdx=areaNameToId[destName];
    if(origIdx===undefined||destIdx===undefined){{alert("Please select valid origin and destination areas.");return;}}
  }}else{{
    if(!pathOriginPoint||!pathDestPoint){{
      alert("Please set both origin and destination points on the map.");
      return;
    }}
    oNode=pathOriginPoint.nodeId;
    dNode=pathDestPoint.nodeId;
  }}

  if(pathLyrGroup)map.removeLayer(pathLyrGroup);

  const k=currentK;
  document.getElementById("pathInfo").innerHTML="<p>Computing path with K="+k+"...</p>";

  setTimeout(()=>{{
    try{{
    let result;
    if(mode==='area'){{
      result=dijkstra(origIdx,destIdx,k);
    }}else{{
      result=dijkstraNodes(oNode,dNode,k);
    }}
    if(result&&result.segments.length>0){{
      // Draw segments with different colors
      pathLyrGroup=L.featureGroup();
      let totalLaneDist=0,totalRoadDist=0,totalUserLaneDist=0;
      for(const seg of result.segments){{
        let coords;
        if(seg.geom){{
          // Wishing lane with full geometry
          coords=seg.geom.map(c=>[c[1],c[0]]);
        }}else{{
          // Regular segment (straight line between nodes)
          coords=[[seg.from[1],seg.from[0]],[seg.to[1],seg.to[0]]];
        }}
        // User lane edges shown in pink, other bike lanes in blue, roads in orange
        let color;
        if(seg.isUserEdge){{
          color="#E91E63"; // Pink for user-drawn lanes
          totalUserLaneDist+=seg.len;
          totalLaneDist+=seg.len;
          console.log('Drawing USER LANE segment (pink):', seg.len.toFixed(0)+'m');
        }}else if(seg.bike){{
          color="#1565C0"; // Blue for existing/wishing bike lanes
          totalLaneDist+=seg.len;
        }}else{{
          color="#E65100"; // Orange for roads
          totalRoadDist+=seg.len;
        }}
        const line=L.polyline(coords,{{color:color,weight:6,opacity:.85}});
        pathLyrGroup.addLayer(line);
      }}
      pathLyrGroup.addTo(map);
      if(pathLyrGroup.getLayers().length>0){{
        map.fitBounds(pathLyrGroup.getBounds(),{{padding:[30,30]}});
      }}

      // Report stats
      const totalDist=totalLaneDist+totalRoadDist;
      const lanePct=totalDist>0?(100*totalLaneDist/totalDist):0;
      const roadPct=totalDist>0?(100*totalRoadDist/totalDist):0;
      const userLanePct=totalDist>0?(100*totalUserLaneDist/totalDist):0;
      const otherLaneDist=totalLaneDist-totalUserLaneDist;
      const otherLanePct=totalDist>0?(100*otherLaneDist/totalDist):0;

      let statsHtml='<div class="path-stats">'+
        '<p><b>Path found</b></p>'+
        '<table>'+
        '<tr><td>Total distance:</td><td>'+(totalDist/1000).toFixed(2)+' km</td></tr>'+
        '<tr><td style="color:#1565C0">On bike lane:</td><td style="color:#1565C0">'+(totalLaneDist/1000).toFixed(2)+' km ('+lanePct.toFixed(1)+'%)</td></tr>';
      if(totalUserLaneDist>0){{
        statsHtml+='<tr><td style="color:#E91E63;padding-left:15px">↳ Your drawn lane:</td><td style="color:#E91E63">'+(totalUserLaneDist/1000).toFixed(2)+' km ('+userLanePct.toFixed(1)+'%)</td></tr>';
      }}
      statsHtml+='<tr><td style="color:#E65100">On road:</td><td style="color:#E65100">'+(totalRoadDist/1000).toFixed(2)+' km ('+roadPct.toFixed(1)+'%)</td></tr>'+
        '<tr><td>Segments:</td><td>'+result.segments.length+'</td></tr>'+
        '</table>'+
        '<div class="bar">';
      if(totalUserLaneDist>0){{
        statsHtml+='<div class="bar-user" style="width:'+userLanePct+'%;background:#E91E63"></div>'+
          '<div class="bar-lane" style="width:'+otherLanePct+'%"></div>';
      }}else{{
        statsHtml+='<div class="bar-lane" style="width:'+lanePct+'%"></div>';
      }}
      statsHtml+='<div class="bar-road" style="width:'+roadPct+'%"></div>'+
        '</div>'+
        '</div>';
      document.getElementById("pathInfo").innerHTML=statsHtml;
    }}else{{
      document.getElementById("pathInfo").innerHTML='<p style="color:#e74c3c">No path found between these areas.</p>';
    }}
    }}catch(err){{
      document.getElementById("pathInfo").innerHTML='<p style="color:#e74c3c">Error computing path: '+err.message+'</p>';
      console.error(err);
    }}
  }},50);
}}

// Build edge lookup: "nodeA_nodeB" -> {{len, bike}}
let edgeLookup=null;
function getEdgeLookup(){{
  if(edgeLookup)return edgeLookup;
  edgeLookup={{}};
  for(const e of EDGES){{
    const a=String(e[0]),b=String(e[1]);
    const info={{len:e[2],bike:!!e[3]}};
    edgeLookup[a+"_"+b]=info;
    edgeLookup[b+"_"+a]=info;
  }}
  return edgeLookup;
}}

function dijkstra(origIdx,destIdx,k){{
  const oc=CENTROIDS[origIdx],dc=CENTROIDS[destIdx];
  let oNode=null,dNode=null,oD=Infinity,dD=Infinity;
  for(const[nid,c]of Object.entries(NODES)){{
    const d1=(c[0]-oc[0])**2+(c[1]-oc[1])**2;
    const d2=(c[0]-dc[0])**2+(c[1]-dc[1])**2;
    if(d1<oD){{oD=d1;oNode=nid;}}
    if(d2<dD){{dD=d2;dNode=nid;}}
  }}
  if(!oNode||!dNode)return null;

  // Build combined edge set from active layers + selected wishing lanes
  const bikeEdgeSet=getAllBikeEdgeSet();
  const virtualEdges=getAllBikeVirtualEdges();

  // Build adjacency with edge info
  const adj=buildAdj(bikeEdgeSet,virtualEdges,k,true);

  const dist={{}},prev={{}},prevEdge={{}},visited=new Set();
  dist[oNode]=0;
  let pq=[[0,oNode]];

  while(pq.length){{
    const[cd,cur]=pq.shift();
    if(visited.has(cur))continue;
    visited.add(cur);
    if(cur===dNode)break;
    for(const{{n,w,len,bike,eKey}}of(adj[cur]||[])){{
      if(visited.has(n))continue;
      const nd=cd+w;
      if(dist[n]===undefined||nd<dist[n]){{
        dist[n]=nd;prev[n]=cur;prevEdge[n]={{len:len,bike:bike,eKey:eKey}};
        let ins=pq.findIndex(x=>x[0]>nd);
        if(ins<0)ins=pq.length;
        pq.splice(ins,0,[nd,n]);
      }}
    }}
  }}

  if(dist[dNode]===undefined)return null;

  // Reconstruct path as segments with edge info
  const segments=[];
  let c=dNode;
  while(prev[c]!==undefined){{
    const p=prev[c];
    const e=prevEdge[c];
    // Use edge geometry if available, otherwise fall back to node coords
    const geom=EDGE_GEOMS[e.eKey];
    if(geom&&geom.length>=2){{
      const fromNode=NODES[p],toNode=NODES[c];
      const g0=geom[0],gN=geom[geom.length-1];
      const d0=Math.abs(g0[0]-fromNode[0])+Math.abs(g0[1]-fromNode[1]);
      const dN=Math.abs(gN[0]-fromNode[0])+Math.abs(gN[1]-fromNode[1]);
      const coords=(d0<=dN)?geom:geom.slice().reverse();
      segments.unshift({{geom:coords,len:e.len,bike:e.bike}});
    }}else{{
      segments.unshift({{from:NODES[p],to:NODES[c],len:e.len,bike:e.bike}});
    }}
    c=p;
  }}
  return {{segments:segments}};
}}

// === ONLINE COMPUTATION ===
function updateComputePanel(){{
  document.getElementById("selCount").textContent=sel.size;
  document.getElementById("compK").textContent=currentK;
  document.getElementById("compT").textContent=currentTheta;
  document.getElementById("compYear").textContent=currentYear;
  // Also update Rank Lanes panel parameters
  document.getElementById("rankK").textContent=currentK;
  document.getElementById("rankT").textContent=currentTheta;
  document.getElementById("rankYear").textContent=currentYear;
}}

// Compute baseline accessibility (active layers only, no wishing lanes) for current K/theta
function computeBaseline(){{
  const k=currentK;
  const theta=currentTheta;

  // Build adjacency list with active layers only (no wishing lanes)
  const bikeEdgeSet=getActiveLayerEdgeSet();
  const virtualEdges=getActiveLayerVirtualEdges();
  const adj=buildAdj(bikeEdgeSet,virtualEdges,k,false);

  const n=AREA_NODES.length;
  const acc_orig=new Array(n).fill(0);
  const acc_dest=new Array(n).fill(0);
  let totalN=0;

  for(let i=0;i<n;i++){{
    const src=String(AREA_NODES[i]);
    const dist={{}},visited=new Set();
    dist[src]=0;
    let pq=[[0,src]];
    while(pq.length){{
      const[cd,cur]=pq.shift();
      if(visited.has(cur))continue;
      visited.add(cur);
      for(const{{n:nb,w}}of(adj[cur]||[])){{
        if(visited.has(nb))continue;
        const nd=cd+w;
        if(dist[nb]===undefined||nd<dist[nb]){{
          dist[nb]=nd;
          let ins=pq.findIndex(x=>x[0]>nd);
          if(ins<0)ins=pq.length;
          pq.splice(ins,0,[nd,nb]);
        }}
      }}
    }}
    for(let j=0;j<n;j++){{
      if(i===j)continue;
      const dstNode=String(AREA_NODES[j]);
      if(dist[dstNode]!==undefined){{
        const tau=Math.max(dist[dstNode]/1000,0.1);
        const decay=Math.pow(tau,theta);
        acc_orig[i]+=destWeight(j)*decay;
        acc_dest[j]+=AREA_POP[i]*decay;
        totalN+=AREA_POP[i]*destWeight(j)*decay;
      }}
    }}
  }}

  baselineAcc={{orig:acc_orig,dest:acc_dest,totalN:totalN}};
  baselineK=k;
  baselineTheta=theta;
  baselineYear=currentYear;
  baselineWEmp=wEmp;
  baselineWEdu=wEdu;
  baselineWTransit=wTransit;
}}

function computeAccessibility(){{
  const k=currentK;
  const theta=currentTheta;
  const selArr=[...sel].sort();
  const selKey=JSON.stringify({{wishing:selArr,layers:activeLayers}});

  const btn=document.getElementById("computeBtn");
  const prog=document.getElementById("computeProgress");
  const results=document.getElementById("computeResults");

  btn.disabled=true;
  btn.textContent="Computing...";

  // First compute baseline if needed
  if(!baselineAcc || baselineK!==k || baselineTheta!==theta || baselineWEmp!==wEmp || baselineWEdu!==wEdu || baselineWTransit!==wTransit){{
    prog.innerHTML="<p>Computing baseline (active layers only)...</p>";
    computeBaseline();
  }}

  prog.innerHTML="<p>Building network with active layers + "+sel.size+" wishing lanes...</p>";

  setTimeout(()=>{{
    // Build combined edge set: active layers + wishing
    const bikeEdgeSet=getAllBikeEdgeSet();
    const virtualEdges=getAllBikeVirtualEdges();
    const adj=buildAdj(bikeEdgeSet,virtualEdges,k,false);

    const n=AREA_NODES.length;
    const acc_orig=new Array(n).fill(0);
    const acc_dest=new Array(n).fill(0);
    let totalN=0;
    let processed=0;

    function processArea(i){{
      if(i>=n){{
        computedAcc={{orig:acc_orig,dest:acc_dest,totalN:totalN}};
        computedK=k;
        computedTheta=theta;
        computedYear=currentYear;
        computedSel=selKey;

        let improvementPct=0;
        let baselineN=0;
        if(baselineAcc && baselineK===k && baselineTheta===theta && baselineYear===currentYear && baselineWEmp===wEmp && baselineWEdu===wEdu && baselineWTransit===wTransit){{
          baselineN=baselineAcc.totalN;
          if(baselineN>0){{
            improvementPct=100*(totalN-baselineN)/baselineN;
          }}
        }}

        btn.disabled=false;
        btn.textContent="Compute Accessibility";
        prog.innerHTML="<p style='color:#27ae60'>Computation complete!</p>";
        results.innerHTML=
          '<div class="path-stats">'+
          '<p><b>Results (K='+k+', &theta;='+theta+', Year='+currentYear+', '+sel.size+' wishing lanes):</b></p>'+
          '<table>'+
          '<tr><td>Baseline N:</td><td>'+baselineN.toExponential(3)+'</td></tr>'+
          '<tr><td>With selected lanes:</td><td>'+totalN.toExponential(3)+'</td></tr>'+
          '<tr><td>Improvement:</td><td style="color:'+(improvementPct>=0?'#27ae60':'#e74c3c')+';font-weight:bold">'+(improvementPct>=0?'+':'')+improvementPct.toFixed(3)+'%</td></tr>'+
          '</table>'+
          '<p style="font-size:.85em;margin-top:8px">Switch to "Change (%)" mode to see per-area improvements.</p>'+
          '</div>';
        updateAreaColors();
        return;
      }}

      const src=String(AREA_NODES[i]);
      const dist={{}},visited=new Set();
      dist[src]=0;
      let pq=[[0,src]];
      while(pq.length){{
        const[cd,cur]=pq.shift();
        if(visited.has(cur))continue;
        visited.add(cur);
        for(const{{n:nb,w}}of(adj[cur]||[])){{
          if(visited.has(nb))continue;
          const nd=cd+w;
          if(dist[nb]===undefined||nd<dist[nb]){{
            dist[nb]=nd;
            let ins=pq.findIndex(x=>x[0]>nd);
            if(ins<0)ins=pq.length;
            pq.splice(ins,0,[nd,nb]);
          }}
        }}
      }}

      for(let j=0;j<n;j++){{
        if(i===j)continue;
        const dstNode=String(AREA_NODES[j]);
        if(dist[dstNode]!==undefined){{
          const tau=Math.max(dist[dstNode]/1000,0.1);
          const decay=Math.pow(tau,theta);
          acc_orig[i]+=destWeight(j)*decay;
          acc_dest[j]+=AREA_POP[i]*decay;
          totalN+=AREA_POP[i]*destWeight(j)*decay;
        }}
      }}

      processed++;
      if(processed%10===0){{
        prog.innerHTML="<p>Processing areas: "+processed+"/"+n+" ("+Math.round(100*processed/n)+"%)</p>";
      }}
      setTimeout(()=>processArea(i+1),0);
    }}

    processArea(0);
  }},50);
}}

// === LANE RANKING ===
function updateRankModeDescription(){{
  const mode=document.querySelector('input[name="rankMode"]:checked').value;
  const desc=document.getElementById('rankModeDescription');
  if(mode==='additive'){{
    desc.innerHTML='<b>Additive mode:</b> Measures the value each lane would add if built next. Baseline is the currently selected lanes.';
  }}else{{
    desc.innerHTML='<b>Subtractive mode:</b> Measures the contribution of each selected lane to the current network. Compares current selection to removing one lane.';
  }}
}}

function computeRankLanes(){{
  const mode=document.querySelector('input[name="rankMode"]:checked').value;
  const k=currentK;
  const theta=currentTheta;

  document.getElementById('rankK').textContent=k;
  document.getElementById('rankT').textContent=theta;
  document.getElementById('rankYear').textContent=currentYear;

  const btn=document.getElementById("rankBtn");
  const prog=document.getElementById("rankProgress");
  const results=document.getElementById("rankResults");

  btn.disabled=true;
  btn.textContent="Computing...";
  prog.innerHTML="<p>Preparing to compute lane rankings...</p>";
  results.innerHTML="";

  setTimeout(()=>rankLanesAsync(mode,k,theta),50);
}}

function rankLanesAsync(mode,k,theta){{
  const prog=document.getElementById("rankProgress");
  const results=document.getElementById("rankResults");
  const btn=document.getElementById("rankBtn");

  // Get all lane IDs
  const allLaneIds=LANE_NAMES.map((_,i)=>i);

  // Determine baseline and test configurations
  let baselineSet,testLanes,testType;
  if(mode==='additive'){{
    // Additive: baseline is current selection, test each unselected lane
    baselineSet=new Set([...sel]);
    testLanes=allLaneIds.filter(id=>!sel.has(id));
    testType='add';
  }}else{{
    // Subtractive: baseline is current selection, test removing each selected lane
    baselineSet=new Set([...sel]);
    testLanes=allLaneIds.filter(id=>sel.has(id));
    testType='remove';
  }}

  if(testLanes.length===0){{
    prog.innerHTML="<p style='color:#e74c3c'>No lanes to test. "+
      (mode==='additive'?"Select fewer lanes or use subtractive mode.":"Select some lanes first.")+"</p>";
    btn.disabled=false;
    btn.textContent="Compute Rankings";
    return;
  }}

  prog.innerHTML="<p>Computing baseline accessibility...</p>";

  // First compute baseline
  const baselineN=computeNetworkValue(baselineSet,k,theta);

  prog.innerHTML="<p>Baseline computed. Testing "+testLanes.length+" lanes...</p>";

  const rankings=[];
  let processed=0;

  function processLane(idx){{
    if(idx>=testLanes.length){{
      // Done - show results sorted by contribution
      rankings.sort((a,b)=>b.contribution-a.contribution);

      let html='<div class="path-stats"><p><b>Lane Rankings ('+mode+' mode, K='+k+', θ='+theta+', Year='+currentYear+'):</b></p>';
      html+='<p style="font-size:.85em">Baseline N: '+baselineN.toExponential(3)+'</p>';
      html+='<table style="width:100%;font-size:.85em">';
      html+='<tr style="background:#e0e0e0"><th>#</th><th>Lane</th><th>Length</th><th>Δ%</th></tr>';

      rankings.forEach((r,i)=>{{
        const color=r.contribution>=0?'#27ae60':'#e74c3c';
        const sign=r.contribution>=0?'+':'';
        const lenKm=(r.length/1000).toFixed(2);
        html+='<tr style="border-bottom:1px solid #ddd">';
        html+='<td style="padding:4px">'+(i+1)+'</td>';
        html+='<td style="padding:4px">'+r.name+'</td>';
        html+='<td style="padding:4px">'+lenKm+' km</td>';
        html+='<td style="padding:4px;color:'+color+';font-weight:bold">'+sign+r.contribution.toFixed(4)+'%</td>';
        html+='</tr>';
      }});

      html+='</table></div>';
      results.innerHTML=html;
      prog.innerHTML="<p style='color:#27ae60'>Ranking complete!</p>";
      btn.disabled=false;
      btn.textContent="Compute Rankings";
      return;
    }}

    const laneId=testLanes[idx];
    const testSet=new Set(baselineSet);

    if(testType==='add'){{
      testSet.add(laneId);
    }}else{{
      testSet.delete(laneId);
    }}

    const testN=computeNetworkValue(testSet,k,theta);

    let contribution;
    if(testType==='add'){{
      // How much does adding this lane improve?
      contribution=baselineN>0?100*(testN-baselineN)/baselineN:0;
    }}else{{
      // How much does removing this lane hurt? (show as positive if lane is valuable)
      contribution=baselineN>0?100*(baselineN-testN)/baselineN:0;
    }}

    rankings.push({{
      id:laneId,
      name:LANE_NAMES[laneId],
      length:LANE_LENGTHS[laneId],
      contribution:contribution,
      value:testN
    }});

    processed++;
    if(processed%3===0||processed===testLanes.length){{
      prog.innerHTML="<p>Testing lanes: "+processed+"/"+testLanes.length+" ("+Math.round(100*processed/testLanes.length)+"%)</p>";
    }}

    setTimeout(()=>processLane(idx+1),0);
  }}

  processLane(0);
}}

function computeNetworkValue(selectedWishingLanes,k,theta){{
  // Build combined edge set: active layers + given wishing lanes
  const bikeEdgeSet=getActiveLayerEdgeSet();
  // Add wishing lane edges
  for(const lid of selectedWishingLanes){{
    const edges=WISHING_EDGES[lid]||[];
    for(const e of edges){{
      const a=Math.min(e[0],e[1]),b=Math.max(e[0],e[1]);
      bikeEdgeSet.add(a+'_'+b);
    }}
  }}
  // Combine virtual edges
  const virtualEdges=[...getActiveLayerVirtualEdges()];
  for(const lid of selectedWishingLanes){{
    const ves=WISHING_VIRTUAL_EDGES[lid]||[];
    for(const ve of ves)virtualEdges.push(ve);
  }}

  const adj=buildAdj(bikeEdgeSet,virtualEdges,k,false);

  const n=AREA_NODES.length;
  let totalN=0;

  for(let i=0;i<n;i++){{
    const src=String(AREA_NODES[i]);
    const dist={{}},visited=new Set();
    dist[src]=0;
    let pq=[[0,src]];

    while(pq.length){{
      const[cd,cur]=pq.shift();
      if(visited.has(cur))continue;
      visited.add(cur);
      for(const{{n:nb,w}}of(adj[cur]||[])){{
        if(visited.has(nb))continue;
        const nd=cd+w;
        if(dist[nb]===undefined||nd<dist[nb]){{
          dist[nb]=nd;
          let ins=pq.findIndex(x=>x[0]>nd);
          if(ins<0)ins=pq.length;
          pq.splice(ins,0,[nd,nb]);
        }}
      }}
    }}

    for(let j=0;j<n;j++){{
      if(i===j)continue;
      const dstNode=String(AREA_NODES[j]);
      if(dist[dstNode]!==undefined){{
        const tau=Math.max(dist[dstNode]/1000,0.1);
        const decay=Math.pow(tau,theta);
        totalN+=AREA_POP[i]*destWeight(j)*decay;
      }}
    }}
  }}

  return totalN;
}}

// === USER LANE DRAWING ===
userLanesLyrGroup=L.featureGroup().addTo(map);

// Haversine distance calculation for lat/lng coordinates
function haversineDistance(lat1,lon1,lat2,lon2){{
  const R=6371000; // Earth radius in meters
  const dLat=(lat2-lat1)*Math.PI/180;
  const dLon=(lon2-lon1)*Math.PI/180;
  const a=Math.sin(dLat/2)*Math.sin(dLat/2)+
    Math.cos(lat1*Math.PI/180)*Math.cos(lat2*Math.PI/180)*
    Math.sin(dLon/2)*Math.sin(dLon/2);
  const c=2*Math.atan2(Math.sqrt(a),Math.sqrt(1-a));
  return R*c;
}}

// Calculate total length of a coordinate array
function calcLaneLength(coords){{
  let total=0;
  for(let i=1;i<coords.length;i++){{
    total+=haversineDistance(coords[i-1][0],coords[i-1][1],coords[i][0],coords[i][1]);
  }}
  return total;
}}

// Find road edges covered by a user-drawn lane (same logic as wishing lanes)
// Uses overlap ratio validation: edge matches if >50% of its length is within buffer
function findCoveredEdges(coords){{
  const covered=new Set();
  const BUFFER_METERS=15; // Same 15m buffer as wishing lanes (was 50m - too large!)
  const OVERLAP_THRESHOLD=0.5; // At least 50% of edge must be covered (same as wishing lanes)
  const SAMPLE_SPACING=5; // Sample points every 5 meters along the edge

  // Helper: get minimum distance from a point to the user lane polyline
  function distToUserLane(lat,lon){{
    let minD=Infinity;
    for(let i=0;i<coords.length-1;i++){{
      const d=pointToSegmentDist(lat,lon,coords[i][0],coords[i][1],coords[i+1][0],coords[i+1][1]);
      if(d<minD)minD=d;
    }}
    return minD;
  }}

  // For each edge in the network, compute overlap ratio
  for(const e of EDGES){{
    const n1=NODES[String(e[0])],n2=NODES[String(e[1])];
    if(!n1||!n2)continue;

    // Edge endpoints in lat/lon (NODES is [lon,lat])
    const e1Lat=n1[1],e1Lon=n1[0];
    const e2Lat=n2[1],e2Lon=n2[0];

    // Calculate edge length
    const edgeLen=haversineDistance(e1Lat,e1Lon,e2Lat,e2Lon);
    if(edgeLen<1)continue; // Skip tiny edges

    // Sample points along the edge and check how many are within buffer of user lane
    const numSamples=Math.max(3,Math.ceil(edgeLen/SAMPLE_SPACING));
    let coveredSamples=0;

    for(let s=0;s<=numSamples;s++){{
      const t=s/numSamples;
      const sampleLat=e1Lat+t*(e2Lat-e1Lat);
      const sampleLon=e1Lon+t*(e2Lon-e1Lon);

      const dist=distToUserLane(sampleLat,sampleLon);
      if(dist<=BUFFER_METERS){{
        coveredSamples++;
      }}
    }}

    // Calculate overlap ratio (what fraction of edge is within buffer)
    const overlapRatio=coveredSamples/(numSamples+1);

    // Match if >50% of edge is covered (same threshold as wishing lanes)
    if(overlapRatio>=OVERLAP_THRESHOLD){{
      const key=Math.min(e[0],e[1])+'_'+Math.max(e[0],e[1]);
      covered.add(key);
    }}
  }}

  console.log('findCoveredEdges: found '+covered.size+' edges for lane with '+coords.length+' points (buffer='+BUFFER_METERS+'m, threshold='+OVERLAP_THRESHOLD*100+'%)');

  // Convert to array of [nodeA,nodeB] pairs
  const result=[];
  for(const key of covered){{
    const[a,b]=key.split('_').map(Number);
    result.push([a,b]);
  }}
  return result;
}}

// Create virtual edges that follow the drawn lane, connecting network nodes
// This allows the lane to create NEW connections in the network
function createVirtualEdges(coords){{
  const CONNECT_THRESHOLD=50; // meters - nodes within this distance can connect to the lane
  const virtualEdges=[];

  // Helper: compute cumulative distance along the lane for a point
  // Returns {{segIdx, t, cumDist}} where segIdx is the segment index, t is position on segment, cumDist is total distance from start
  function projectToLane(lat,lng){{
    let bestDist=Infinity;
    let bestSegIdx=0;
    let bestT=0;

    for(let i=0;i<coords.length-1;i++){{
      const x1=coords[i][0],y1=coords[i][1];
      const x2=coords[i+1][0],y2=coords[i+1][1];
      const dx=x2-x1,dy=y2-y1;
      let t=0;
      if(dx!==0||dy!==0){{
        t=Math.max(0,Math.min(1,((lat-x1)*dx+(lng-y1)*dy)/(dx*dx+dy*dy)));
      }}
      const projLat=x1+t*dx;
      const projLng=y1+t*dy;
      const d=haversineDistance(lat,lng,projLat,projLng);
      if(d<bestDist){{
        bestDist=d;
        bestSegIdx=i;
        bestT=t;
      }}
    }}

    // Compute cumulative distance from start to the projection point
    let cumDist=0;
    for(let i=0;i<bestSegIdx;i++){{
      cumDist+=haversineDistance(coords[i][0],coords[i][1],coords[i+1][0],coords[i+1][1]);
    }}
    // Add partial distance on the current segment
    const segLen=haversineDistance(coords[bestSegIdx][0],coords[bestSegIdx][1],coords[bestSegIdx+1][0],coords[bestSegIdx+1][1]);
    cumDist+=bestT*segLen;

    return {{dist:bestDist,segIdx:bestSegIdx,t:bestT,cumDist:cumDist}};
  }}

  // Find all network nodes within threshold of the lane
  const nearbyNodes=[];
  for(const nodeId in NODES){{
    const node=NODES[nodeId];
    const nodeLat=node[1],nodeLon=node[0];

    // Quick bounding box check first
    let minLat=Infinity,maxLat=-Infinity,minLng=Infinity,maxLng=-Infinity;
    for(const c of coords){{
      if(c[0]<minLat)minLat=c[0];
      if(c[0]>maxLat)maxLat=c[0];
      if(c[1]<minLng)minLng=c[1];
      if(c[1]>maxLng)maxLng=c[1];
    }}
    const margin=CONNECT_THRESHOLD/111000; // rough degrees
    if(nodeLat<minLat-margin||nodeLat>maxLat+margin||nodeLon<minLng-margin||nodeLon>maxLng+margin)continue;

    const proj=projectToLane(nodeLat,nodeLon);
    if(proj.dist<=CONNECT_THRESHOLD){{
      nearbyNodes.push({{
        nodeId:parseInt(nodeId),
        lat:nodeLat,
        lng:nodeLon,
        cumDist:proj.cumDist,
        projDist:proj.dist,
        segIdx:proj.segIdx,
        t:proj.t
      }});
    }}
  }}

  // Sort by cumulative distance along the lane
  nearbyNodes.sort((a,b)=>a.cumDist-b.cumDist);

  console.log('createVirtualEdges: found '+nearbyNodes.length+' nodes within '+CONNECT_THRESHOLD+'m of lane');

  // Helper: interpolate a point along a segment
  function interpolatePoint(segIdx,t){{
    const p1=coords[segIdx];
    const p2=coords[segIdx+1];
    return [p1[0]+t*(p2[0]-p1[0]),p1[1]+t*(p2[1]-p1[1])];
  }}

  // Helper: extract geometry between two projection points along the lane
  function extractGeometry(n1,n2){{
    const geom=[];
    // Start point: interpolated position on n1's segment
    geom.push(interpolatePoint(n1.segIdx,n1.t));
    // Add intermediate coords between the two nodes
    for(let seg=n1.segIdx;seg<=n2.segIdx;seg++){{
      if(seg===n1.segIdx){{
        // For the first segment, add the end point if we're not at the end
        if(n1.t<1&&seg<n2.segIdx){{
          geom.push([coords[seg+1][0],coords[seg+1][1]]);
        }}
      }}else if(seg===n2.segIdx){{
        // For the last segment, we'll add the interpolated end point below
      }}else{{
        // For intermediate segments, add the end point
        geom.push([coords[seg+1][0],coords[seg+1][1]]);
      }}
    }}
    // End point: interpolated position on n2's segment
    geom.push(interpolatePoint(n2.segIdx,n2.t));
    // Convert to [lon,lat] format to match EDGE_GEOMS
    return geom.map(p=>[p[1],p[0]]);
  }}

  // Create virtual edges between consecutive nodes along the lane
  for(let i=0;i<nearbyNodes.length-1;i++){{
    const n1=nearbyNodes[i];
    const n2=nearbyNodes[i+1];

    // Edge length is the distance along the lane between projections
    const edgeLen=n2.cumDist-n1.cumDist;

    // Only create edge if there's meaningful distance (avoid 0-length edges)
    if(edgeLen>5){{
      // Extract the actual lane geometry for this edge
      const geom=extractGeometry(n1,n2);
      virtualEdges.push({{
        from:n1.nodeId,
        to:n2.nodeId,
        len:edgeLen,
        // Store the lane path segment for visualization
        geometry:geom
      }});
    }}
  }}

  console.log('createVirtualEdges: created '+virtualEdges.length+' virtual edges');
  return virtualEdges;
}}

// Point to line segment distance (approximate in lat/lng)
function pointToSegmentDist(px,py,x1,y1,x2,y2){{
  const dx=x2-x1,dy=y2-y1;
  if(dx===0&&dy===0)return haversineDistance(px,py,x1,y1);
  const t=Math.max(0,Math.min(1,((px-x1)*dx+(py-y1)*dy)/(dx*dx+dy*dy)));
  const projX=x1+t*dx,projY=y1+t*dy;
  return haversineDistance(px,py,projX,projY);
}}

// Snap threshold in meters (same as road edge matching)
const SNAP_THRESHOLD=15;

// Find the nearest snap point (network node or user lane point) within threshold
function findNearestSnapPoint(lat,lng){{
  let bestDist=Infinity;
  let bestPoint=null;
  let bestType=null;

  // Check all network nodes
  for(const nodeId in NODES){{
    const node=NODES[nodeId];
    // NODES stores [lon, lat]
    const nodeLat=node[1],nodeLon=node[0];
    const d=haversineDistance(lat,lng,nodeLat,nodeLon);
    if(d<bestDist){{
      bestDist=d;
      bestPoint={{lat:nodeLat,lng:nodeLon}};
      bestType='node';
    }}
  }}

  // Check all points from existing user lanes
  for(const lane of userLanes){{
    for(const coord of lane.coords){{
      // userLanes coords are [lat, lng]
      const d=haversineDistance(lat,lng,coord[0],coord[1]);
      if(d<bestDist){{
        bestDist=d;
        bestPoint={{lat:coord[0],lng:coord[1]}};
        bestType='lane';
      }}
    }}
  }}

  // Return snapped point if within threshold
  if(bestDist<=SNAP_THRESHOLD&&bestPoint){{
    return{{lat:bestPoint.lat,lng:bestPoint.lng,snapped:true,type:bestType}};
  }}
  return{{lat:lat,lng:lng,snapped:false}};
}}

// Project a point onto a line segment, returns {{t, dist}}
function projectPointToSegment(px,py,x1,y1,x2,y2){{
  const dx=x2-x1,dy=y2-y1;
  if(dx===0&&dy===0)return{{t:0,dist:haversineDistance(px,py,x1,y1)}};
  const t=Math.max(0,Math.min(1,((px-x1)*dx+(py-y1)*dy)/(dx*dx+dy*dy)));
  const projX=x1+t*dx,projY=y1+t*dy;
  return{{t:t,dist:haversineDistance(px,py,projX,projY)}};
}}

// Insert connection points where the lane passes near road nodes
// This ensures the lane connects to the road network at intersections
function insertConnectionPoints(coords){{
  const result=[];
  const threshold=SNAP_THRESHOLD;

  for(let i=0;i<coords.length-1;i++){{
    const p1=coords[i],p2=coords[i+1];
    result.push(p1);

    // Collect nearby points to insert along this segment
    const toInsert=[];

    // Check network nodes
    for(const nodeId in NODES){{
      const node=NODES[nodeId];
      const nodeLat=node[1],nodeLon=node[0];
      const proj=projectPointToSegment(nodeLat,nodeLon,p1[0],p1[1],p2[0],p2[1]);

      // Only insert if close enough and not at segment endpoints
      if(proj.dist<=threshold&&proj.t>0.05&&proj.t<0.95){{
        toInsert.push({{t:proj.t,lat:nodeLat,lng:nodeLon,type:'node'}});
      }}
    }}

    // Check existing user lane points
    for(const lane of userLanes){{
      for(const coord of lane.coords){{
        const proj=projectPointToSegment(coord[0],coord[1],p1[0],p1[1],p2[0],p2[1]);
        if(proj.dist<=threshold&&proj.t>0.05&&proj.t<0.95){{
          toInsert.push({{t:proj.t,lat:coord[0],lng:coord[1],type:'lane'}});
        }}
      }}
    }}

    // Sort by t and insert in order, removing duplicates (within 5m)
    toInsert.sort((a,b)=>a.t-b.t);
    let lastInserted=null;
    for(const pt of toInsert){{
      if(lastInserted&&haversineDistance(pt.lat,pt.lng,lastInserted[0],lastInserted[1])<5){{
        continue;
      }}
      result.push([pt.lat,pt.lng]);
      lastInserted=[pt.lat,pt.lng];
    }}
  }}

  // Add the last point
  result.push(coords[coords.length-1]);
  return result;
}}

// Markers for snapped points during drawing
let snapMarkers=[];

function clearSnapMarkers(){{
  for(const m of snapMarkers){{
    map.removeLayer(m);
  }}
  snapMarkers=[];
}}

function startDrawing(){{
  if(isDrawing)return;
  isDrawing=true;

  document.getElementById('drawBtn').style.display='none';
  document.getElementById('cancelBtn').style.display='inline';
  document.getElementById('finishBtn').style.display='inline';
  document.getElementById('drawingStatus').innerHTML='<b style="color:#e91e63">Drawing mode active.</b> Click on the map to add points. Double-click or press Finish to complete.';

  // Create a new polyline for drawing
  currentDrawLayer=L.polyline([],{{
    color:'#E91E63',
    weight:5,
    opacity:0.8,
    dashArray:'10,5'
  }}).addTo(map);

  // Add click handler for drawing
  map.on('click',onDrawClick);
  map.on('dblclick',finishDrawing);

  // Change cursor
  map.getContainer().style.cursor='crosshair';
}}

function onDrawClick(e){{
  if(!isDrawing||!currentDrawLayer)return;

  // Try to snap the clicked point to nearby network nodes or user lane points
  const snap=findNearestSnapPoint(e.latlng.lat,e.latlng.lng);
  const latlng=L.latLng(snap.lat,snap.lng);

  const coords=currentDrawLayer.getLatLngs();
  coords.push(latlng);
  currentDrawLayer.setLatLngs(coords);

  // Add visual marker for snapped points
  if(snap.snapped){{
    const marker=L.circleMarker(latlng,{{
      radius:6,
      color:'#27ae60',
      fillColor:'#2ecc71',
      fillOpacity:0.8,
      weight:2
    }}).addTo(map);
    marker.bindTooltip('Snapped to '+(snap.type==='node'?'road':'lane'),{{permanent:false,direction:'top'}});
    snapMarkers.push(marker);
  }}

  const len=calcLaneLength(coords.map(c=>[c.lat,c.lng]));
  const snapStatus=snap.snapped?' <span style="color:#27ae60">(snapped to '+snap.type+')</span>':'';
  document.getElementById('drawingStatus').innerHTML=
    '<b style="color:#e91e63">Drawing...</b> Points: '+coords.length+', Length: '+(len/1000).toFixed(2)+' km.'+snapStatus+' Double-click or Finish to complete.';
}}

function cancelDrawing(){{
  if(!isDrawing)return;
  isDrawing=false;

  if(currentDrawLayer){{
    map.removeLayer(currentDrawLayer);
    currentDrawLayer=null;
  }}
  clearSnapMarkers();

  map.off('click',onDrawClick);
  map.off('dblclick',finishDrawing);
  map.getContainer().style.cursor='';

  document.getElementById('drawBtn').style.display='inline';
  document.getElementById('cancelBtn').style.display='none';
  document.getElementById('finishBtn').style.display='none';
  document.getElementById('drawingStatus').innerHTML='Drawing cancelled.';
}}

function finishDrawing(){{
  if(!isDrawing||!currentDrawLayer)return;

  const coords=currentDrawLayer.getLatLngs();
  if(coords.length<2){{
    alert('Please draw at least 2 points to create a lane.');
    return;
  }}

  isDrawing=false;
  map.off('click',onDrawClick);
  map.off('dblclick',finishDrawing);
  map.getContainer().style.cursor='';

  // Remove temporary dashed layer and snap markers
  map.removeLayer(currentDrawLayer);
  currentDrawLayer=null;
  clearSnapMarkers();

  document.getElementById('drawBtn').style.display='inline';
  document.getElementById('cancelBtn').style.display='none';
  document.getElementById('finishBtn').style.display='none';

  // Prompt for lane name
  const defaultName='Custom Lane '+(userLaneIdCounter+1);
  const name=prompt('Enter a name for this lane:',defaultName)||defaultName;

  // Create the lane object with connection points inserted at intersections
  let coordsArr=coords.map(c=>[c.lat,c.lng]);
  const origLen=coordsArr.length;
  coordsArr=insertConnectionPoints(coordsArr);
  const insertedCount=coordsArr.length-origLen;
  console.log('finishDrawing: inserted '+insertedCount+' connection points');

  const length=calcLaneLength(coordsArr);
  const edges=findCoveredEdges(coordsArr);
  const virtualEdges=createVirtualEdges(coordsArr);

  const lane={{
    id:userLaneIdCounter++,
    name:name,
    coords:coordsArr,
    length:length,
    active:true,
    layer:null,
    edges:edges,
    virtualEdges:virtualEdges
  }};

  // Create the visual layer
  lane.layer=L.polyline(coords,{{
    color:'#E91E63',
    weight:5,
    opacity:0.85
  }});
  lane.layer.bindPopup('<b>'+name+'</b><br>Length: '+(length/1000).toFixed(2)+' km<br>Matched edges: '+edges.length+'<br>Virtual edges: '+virtualEdges.length+'<br>Connection points: '+insertedCount);
  lane.layer.on('click',function(e){{
    // Don't stop propagation during drawing mode
    if(isDrawing)return;
    L.DomEvent.stopPropagation(e);
  }});
  userLanesLyrGroup.addLayer(lane.layer);

  userLanes.push(lane);

  const connMsg=insertedCount>0?' ('+insertedCount+' connection pts)':'';
  const virtualMsg=virtualEdges.length>0?', '+virtualEdges.length+' virtual edges':'';
  document.getElementById('drawingStatus').innerHTML='<span style="color:#27ae60">Lane "'+name+'" created! ('+edges.length+' road edges'+virtualMsg+connMsg+')</span>';

  buildUserLaneList();
  invalidateUserLaneComputation();
}}

function buildUserLaneList(){{
  document.getElementById('userLaneCount').textContent=userLanes.length;
  const container=document.getElementById('userLaneList');
  if(userLanes.length===0){{
    container.innerHTML='<p style="color:#999;font-size:.85em">No custom lanes drawn yet. Use "Start Drawing" to create one.</p>';
    return;
  }}

  container.innerHTML=userLanes.map(lane=>{{
    const activeClass=lane.active?'active':'';
    const toggleBtn=lane.active?'Disable':'Enable';
    return '<div class="user-lane '+activeClass+'" id="ulane-'+lane.id+'">'+
      '<div class="lane-header">'+
      '<span class="lane-name">'+lane.name+'</span>'+
      '<span class="lane-length">'+(lane.length/1000).toFixed(2)+' km</span>'+
      '</div>'+
      '<div class="lane-actions">'+
      '<button onclick="toggleUserLane('+lane.id+')">'+toggleBtn+'</button>'+
      '<button onclick="zoomToUserLane('+lane.id+')">Zoom</button>'+
      '<button onclick="renameUserLane('+lane.id+')">Rename</button>'+
      '<button class="del" onclick="deleteUserLane('+lane.id+')">Delete</button>'+
      '</div>'+
      '<div style="font-size:.8em;color:#666;margin-top:4px">'+lane.edges.length+' road edges matched</div>'+
      '</div>';
  }}).join('');
}}

function toggleUserLane(id){{
  const lane=userLanes.find(l=>l.id===id);
  if(!lane)return;
  lane.active=!lane.active;

  if(lane.active){{
    lane.layer.setStyle({{opacity:0.85}});
    userLanesLyrGroup.addLayer(lane.layer);
  }}else{{
    lane.layer.setStyle({{opacity:0.3}});
  }}

  buildUserLaneList();
  invalidateUserLaneComputation();
}}

function zoomToUserLane(id){{
  const lane=userLanes.find(l=>l.id===id);
  if(!lane||!lane.layer)return;
  map.fitBounds(lane.layer.getBounds(),{{padding:[50,50]}});
}}

function renameUserLane(id){{
  const lane=userLanes.find(l=>l.id===id);
  if(!lane)return;
  const newName=prompt('Enter new name:',lane.name);
  if(newName&&newName.trim()){{
    lane.name=newName.trim();
    lane.layer.setPopupContent('<b>'+lane.name+'</b><br>Length: '+(lane.length/1000).toFixed(2)+' km<br>Matched edges: '+lane.edges.length);
    buildUserLaneList();
  }}
}}

function deleteUserLane(id){{
  const idx=userLanes.findIndex(l=>l.id===id);
  if(idx<0)return;
  if(!confirm('Delete lane "'+userLanes[idx].name+'"?'))return;

  const lane=userLanes[idx];
  if(lane.layer){{
    userLanesLyrGroup.removeLayer(lane.layer);
  }}
  userLanes.splice(idx,1);
  buildUserLaneList();
  invalidateUserLaneComputation();
}}

function invalidateUserLaneComputation(){{
  // Clear computed results when user lanes change
  computedAcc=null;
  computedK=null;
  computedTheta=null;
  computedYear=null;
  computedSel=null;
  updateAreaColors();
}}

// === EXPORT/IMPORT USER LANES ===
function exportUserLanes(){{
  if(userLanes.length===0){{
    alert('No lanes to export. Draw some lanes first!');
    return;
  }}

  const features=userLanes.map(lane=>({{
    type:'Feature',
    properties:{{
      name:lane.name,
      length_m:Math.round(lane.length),
      active:lane.active
    }},
    geometry:{{
      type:'LineString',
      coordinates:lane.coords.map(c=>[c[1],c[0]]) // GeoJSON uses [lon,lat]
    }}
  }}));

  const geojson={{
    type:'FeatureCollection',
    features:features
  }};

  const blob=new Blob([JSON.stringify(geojson,null,2)],{{type:'application/json'}});
  const url=URL.createObjectURL(blob);
  const a=document.createElement('a');
  a.href=url;
  a.download='custom_bike_lanes.geojson';
  a.click();
  URL.revokeObjectURL(url);
}}

function importUserLanes(event){{
  const file=event.target.files[0];
  if(!file)return;

  const reader=new FileReader();
  reader.onload=function(e){{
    try{{
      const geojson=JSON.parse(e.target.result);
      if(!geojson.features||!Array.isArray(geojson.features)){{
        throw new Error('Invalid GeoJSON: missing features array');
      }}

      let imported=0;
      for(const feat of geojson.features){{
        if(feat.geometry?.type!=='LineString')continue;

        const coords=feat.geometry.coordinates.map(c=>[c[1],c[0]]); // Convert [lon,lat] to [lat,lon]
        if(coords.length<2)continue;

        const name=feat.properties?.name||('Imported Lane '+(userLaneIdCounter+1));
        const length=calcLaneLength(coords);
        const edges=findCoveredEdges(coords);
        const virtualEdges=createVirtualEdges(coords);

        const lane={{
          id:userLaneIdCounter++,
          name:name,
          coords:coords,
          length:length,
          active:feat.properties?.active!==false,
          layer:null,
          edges:edges,
          virtualEdges:virtualEdges
        }};

        lane.layer=L.polyline(coords.map(c=>[c[0],c[1]]),{{
          color:'#E91E63',
          weight:5,
          opacity:lane.active?0.85:0.3
        }});
        lane.layer.bindPopup('<b>'+lane.name+'</b><br>Length: '+(length/1000).toFixed(2)+' km<br>Matched edges: '+edges.length+'<br>Virtual edges: '+virtualEdges.length);
        userLanesLyrGroup.addLayer(lane.layer);
        userLanes.push(lane);
        imported++;
      }}

      buildUserLaneList();
      invalidateUserLaneComputation();
      alert('Imported '+imported+' lane(s) successfully!');
    }}catch(err){{
      alert('Error importing GeoJSON: '+err.message);
    }}
  }};
  reader.readAsText(file);
  event.target.value=''; // Reset file input
}}

// Modify dijkstra to include user lanes
const originalDijkstra=dijkstra;
dijkstra=function(origIdx,destIdx,k){{
  // Get edges covered by active user lanes
  const userEdgeSet=new Set();
  for(const lane of userLanes){{
    if(!lane.active)continue;
    for(const e of lane.edges){{
      const key=Math.min(e[0],e[1])+'_'+Math.max(e[0],e[1]);
      userEdgeSet.add(key);
    }}
  }}

  // If no user lanes, use original
  if(userEdgeSet.size===0){{
    return originalDijkstra(origIdx,destIdx,k);
  }}

  const oc=CENTROIDS[origIdx],dc=CENTROIDS[destIdx];
  let oNode=null,dNode=null,oD=Infinity,dD=Infinity;
  for(const[nid,c]of Object.entries(NODES)){{
    const d1=(c[0]-oc[0])**2+(c[1]-oc[1])**2;
    const d2=(c[0]-dc[0])**2+(c[1]-dc[1])**2;
    if(d1<oD){{oD=d1;oNode=nid;}}
    if(d2<dD){{dD=d2;dNode=nid;}}
  }}
  if(!oNode||!dNode)return null;

  // Build combined edge set: active layers + wishing + user lanes
  const bikeEdgeSet=getAllBikeEdgeSet();
  for(const key of userEdgeSet)bikeEdgeSet.add(key);

  // Build adjacency with user edge tracking
  const adj={{}};
  for(const e of EDGES){{
    const len=e[2];
    const a=String(e[0]),b=String(e[1]);
    const edgeKey=Math.min(e[0],e[1])+'_'+Math.max(e[0],e[1]);
    const isUserEdge=userEdgeSet.has(edgeKey);
    const bike=bikeEdgeSet.has(edgeKey);
    const discomfort=e[4]||1.0;
    const w=bike?len:len*k*discomfort;
    if(!adj[a])adj[a]=[];
    if(!adj[b])adj[b]=[];
    adj[a].push({{n:b,w:w,len:len,bike:bike,eKey:edgeKey,isUserEdge:isUserEdge}});
    adj[b].push({{n:a,w:w,len:len,bike:bike,eKey:edgeKey,isUserEdge:isUserEdge}});
  }}

  // Add virtual edges from all sources (layers + wishing + user)
  const layerVEs=getAllBikeVirtualEdges();
  for(const ve of layerVEs){{
    const a=String(ve.from||ve['from']),b=String(ve.to||ve['to']);
    const len=ve.len;
    const eKey='ve_'+a+'_'+b;
    if(!adj[a])adj[a]=[];
    if(!adj[b])adj[b]=[];
    adj[a].push({{n:b,w:len,len:len,bike:true,eKey:eKey}});
    adj[b].push({{n:a,w:len,len:len,bike:true,eKey:eKey}});
  }}

  const virtualEdgeGeoms={{}};
  for(const lane of userLanes){{
    if(!lane.active||!lane.virtualEdges)continue;
    for(const ve of lane.virtualEdges){{
      const a=String(ve.from),b=String(ve.to);
      const len=ve.len;
      const eKey='virtual_'+lane.id+'_'+ve.from+'_'+ve.to;
      if(ve.geometry)virtualEdgeGeoms[eKey]=ve.geometry;
      if(!adj[a])adj[a]=[];
      if(!adj[b])adj[b]=[];
      adj[a].push({{n:b,w:len,len:len,bike:true,eKey:eKey,isUserEdge:true,isVirtual:true}});
      adj[b].push({{n:a,w:len,len:len,bike:true,eKey:eKey,isUserEdge:true,isVirtual:true}});
    }}
  }}

  const dist={{}},prev={{}},prevEdge={{}},visited=new Set();
  dist[oNode]=0;
  let pq=[[0,oNode]];

  while(pq.length){{
    const[cd,cur]=pq.shift();
    if(visited.has(cur))continue;
    visited.add(cur);
    if(cur===dNode)break;
    for(const{{n,w,len,bike,eKey,isUserEdge}}of(adj[cur]||[])){{
      if(visited.has(n))continue;
      const nd=cd+w;
      if(dist[n]===undefined||nd<dist[n]){{
        dist[n]=nd;prev[n]=cur;prevEdge[n]={{len:len,bike:bike,eKey:eKey,isUserEdge:isUserEdge}};
        let ins=pq.findIndex(x=>x[0]>nd);
        if(ins<0)ins=pq.length;
        pq.splice(ins,0,[nd,n]);
      }}
    }}
  }}

  if(dist[dNode]===undefined)return null;

  const segments=[];
  let c=dNode;
  while(prev[c]!==undefined){{
    const p=prev[c];
    const e=prevEdge[c];
    let geom=EDGE_GEOMS[e.eKey];
    if(!geom&&virtualEdgeGeoms[e.eKey])geom=virtualEdgeGeoms[e.eKey];
    if(geom&&geom.length>=2){{
      const fromNode=NODES[p],toNode=NODES[c];
      const g0=geom[0],gN=geom[geom.length-1];
      const d0=Math.abs(g0[0]-fromNode[0])+Math.abs(g0[1]-fromNode[1]);
      const dN=Math.abs(gN[0]-fromNode[0])+Math.abs(gN[1]-fromNode[1]);
      const coords=(d0<=dN)?geom:geom.slice().reverse();
      segments.unshift({{geom:coords,len:e.len,bike:e.bike,isUserEdge:e.isUserEdge}});
    }}else{{
      segments.unshift({{from:NODES[p],to:NODES[c],len:e.len,bike:e.bike,isUserEdge:e.isUserEdge}});
    }}
    c=p;
  }}
  return {{segments:segments}};
}};

// Dijkstra with node IDs directly (for map point selection)
function dijkstraNodes(oNode,dNode,k){{
  // Get edges covered by active user lanes
  const userEdgeSet=new Set();
  for(const lane of userLanes){{
    if(!lane.active)continue;
    for(const e of lane.edges){{
      const key=Math.min(e[0],e[1])+'_'+Math.max(e[0],e[1]);
      userEdgeSet.add(key);
    }}
  }}

  // Build combined edge set: active layers + wishing + user lanes
  const bikeEdgeSet=getAllBikeEdgeSet();
  for(const key of userEdgeSet)bikeEdgeSet.add(key);

  // Build adjacency with user edge tracking
  const adj={{}};
  for(const e of EDGES){{
    const len=e[2];
    const a=String(e[0]),b=String(e[1]);
    const edgeKey=Math.min(e[0],e[1])+'_'+Math.max(e[0],e[1]);
    const isUserEdge=userEdgeSet.has(edgeKey);
    const bike=bikeEdgeSet.has(edgeKey);
    const discomfort=e[4]||1.0;
    const w=bike?len:len*k*discomfort;
    if(!adj[a])adj[a]=[];
    if(!adj[b])adj[b]=[];
    adj[a].push({{n:b,w:w,len:len,bike:bike,eKey:edgeKey,isUserEdge:isUserEdge}});
    adj[b].push({{n:a,w:w,len:len,bike:bike,eKey:edgeKey,isUserEdge:isUserEdge}});
  }}

  // Add virtual edges from all sources
  const layerVEs=getAllBikeVirtualEdges();
  for(const ve of layerVEs){{
    const a=String(ve.from||ve['from']),b=String(ve.to||ve['to']);
    const len=ve.len;
    const eKey='ve_'+a+'_'+b;
    if(!adj[a])adj[a]=[];
    if(!adj[b])adj[b]=[];
    adj[a].push({{n:b,w:len,len:len,bike:true,eKey:eKey}});
    adj[b].push({{n:a,w:len,len:len,bike:true,eKey:eKey}});
  }}

  const virtualEdgeGeoms={{}};
  for(const lane of userLanes){{
    if(!lane.active||!lane.virtualEdges)continue;
    for(const ve of lane.virtualEdges){{
      const a=String(ve.from),b=String(ve.to);
      const len=ve.len;
      const eKey='virtual_'+lane.id+'_'+ve.from+'_'+ve.to;
      if(ve.geometry)virtualEdgeGeoms[eKey]=ve.geometry;
      if(!adj[a])adj[a]=[];
      if(!adj[b])adj[b]=[];
      adj[a].push({{n:b,w:len,len:len,bike:true,eKey:eKey,isUserEdge:true,isVirtual:true}});
      adj[b].push({{n:a,w:len,len:len,bike:true,eKey:eKey,isUserEdge:true,isVirtual:true}});
    }}
  }}

  const dist={{}},prev={{}},prevEdge={{}},visited=new Set();
  dist[oNode]=0;
  let pq=[[0,oNode]];

  while(pq.length){{
    const[cd,cur]=pq.shift();
    if(visited.has(cur))continue;
    visited.add(cur);
    if(cur===dNode)break;
    for(const{{n,w,len,bike,eKey,isUserEdge}}of(adj[cur]||[])){{
      if(visited.has(n))continue;
      const nd=cd+w;
      if(dist[n]===undefined||nd<dist[n]){{
        dist[n]=nd;prev[n]=cur;prevEdge[n]={{len:len,bike:bike,eKey:eKey,isUserEdge:isUserEdge}};
        let ins=pq.findIndex(x=>x[0]>nd);
        if(ins<0)ins=pq.length;
        pq.splice(ins,0,[nd,n]);
      }}
    }}
  }}

  if(dist[dNode]===undefined)return null;

  const segments=[];
  let c=dNode;
  while(prev[c]!==undefined){{
    const p=prev[c];
    const e=prevEdge[c];
    let geom=EDGE_GEOMS[e.eKey];
    if(!geom&&virtualEdgeGeoms[e.eKey])geom=virtualEdgeGeoms[e.eKey];
    if(geom&&geom.length>=2){{
      const fromNode=NODES[p],toNode=NODES[c];
      const g0=geom[0],gN=geom[geom.length-1];
      const d0=Math.abs(g0[0]-fromNode[0])+Math.abs(g0[1]-fromNode[1]);
      const dN=Math.abs(gN[0]-fromNode[0])+Math.abs(gN[1]-fromNode[1]);
      const coords=(d0<=dN)?geom:geom.slice().reverse();
      segments.unshift({{geom:coords,len:e.len,bike:e.bike,isUserEdge:e.isUserEdge}});
    }}else{{
      segments.unshift({{from:NODES[p],to:NODES[c],len:e.len,bike:e.bike,isUserEdge:e.isUserEdge}});
    }}
    c=p;
  }}
  return {{segments:segments}};
}}

// Modify computeAccessibility to include user lanes
const originalComputeAccessibility=computeAccessibility;
computeAccessibility=function(){{
  const k=currentK;
  const theta=currentTheta;
  const selArr=[...sel].sort();

  const userActiveIds=userLanes.filter(l=>l.active).map(l=>l.id).sort();
  const selKey=JSON.stringify({{wishing:selArr,user:userActiveIds,layers:activeLayers}});

  const btn=document.getElementById("computeBtn");
  const prog=document.getElementById("computeProgress");
  const results=document.getElementById("computeResults");

  btn.disabled=true;
  btn.textContent="Computing...";

  // First compute baseline if needed (active layers only, no wishing)
  if(!baselineAcc || baselineK!==k || baselineTheta!==theta || baselineWEmp!==wEmp || baselineWEdu!==wEdu || baselineWTransit!==wTransit){{
    prog.innerHTML="<p>Computing baseline (active layers only, no wishing)...</p>";
    computeBaseline();
  }}

  prog.innerHTML="<p>Building network with active layers + "+sel.size+" wishing lanes + "+userActiveIds.length+" custom lanes...</p>";

  setTimeout(()=>{{
    // Build combined edge set: active layers + wishing + user lanes
    const bikeEdgeSet=getAllBikeEdgeSet();
    for(const lane of userLanes){{
      if(!lane.active)continue;
      for(const e of lane.edges){{
        const key=Math.min(e[0],e[1])+'_'+Math.max(e[0],e[1]);
        bikeEdgeSet.add(key);
      }}
    }}

    // Combine virtual edges
    const virtualEdges=getAllBikeVirtualEdges();
    for(const lane of userLanes){{
      if(!lane.active||!lane.virtualEdges)continue;
      for(const ve of lane.virtualEdges)virtualEdges.push(ve);
    }}

    const adj=buildAdj(bikeEdgeSet,virtualEdges,k,false);

    const n=AREA_NODES.length;
    const acc_orig=new Array(n).fill(0);
    const acc_dest=new Array(n).fill(0);
    let totalN=0;
    let processed=0;

    function processArea(i){{
      if(i>=n){{
        computedAcc={{orig:acc_orig,dest:acc_dest,totalN:totalN}};
        computedK=k;
        computedTheta=theta;
        computedYear=currentYear;
        computedSel=selKey;

        let improvementPct=0;
        let baselineN=0;
        if(baselineAcc && baselineK===k && baselineTheta===theta && baselineYear===currentYear && baselineWEmp===wEmp && baselineWEdu===wEdu && baselineWTransit===wTransit){{
          baselineN=baselineAcc.totalN;
          if(baselineN>0){{
            improvementPct=100*(totalN-baselineN)/baselineN;
          }}
        }}

        btn.disabled=false;
        btn.textContent="Compute Accessibility";
        prog.innerHTML="<p style='color:#27ae60'>Computation complete!</p>";

        const userCount=userLanes.filter(l=>l.active).length;
        results.innerHTML=
          '<div class="path-stats">'+
          '<p><b>Results (K='+k+', &theta;='+theta+', Year='+currentYear+'):</b></p>'+
          '<p style="font-size:.85em">'+sel.size+' wishing lanes + '+userCount+' custom lanes</p>'+
          '<table>'+
          '<tr><td>Baseline N:</td><td>'+baselineN.toExponential(3)+'</td></tr>'+
          '<tr><td>With selected lanes:</td><td>'+totalN.toExponential(3)+'</td></tr>'+
          '<tr><td>Improvement:</td><td style="color:'+(improvementPct>=0?'#27ae60':'#e74c3c')+';font-weight:bold">'+(improvementPct>=0?'+':'')+improvementPct.toFixed(3)+'%</td></tr>'+
          '</table>'+
          '<p style="font-size:.85em;margin-top:8px">Switch to "Change (%)" mode to see per-area improvements.</p>'+
          '</div>';
        updateAreaColors();

        if(userCount>0){{
          document.getElementById('userLaneImpact').innerHTML=
            '<div class="impact-box'+(improvementPct<0?' negative':'')+'">'+
            '<h4>Impact of Custom Lanes</h4>'+
            '<p>Your '+userCount+' custom lane(s) '+
            (improvementPct>=0?'improve':'reduce')+' accessibility by <b>'+(improvementPct>=0?'+':'')+improvementPct.toFixed(3)+'%</b></p>'+
            '<p style="font-size:.85em;color:#666">(Combined with '+sel.size+' selected wishing lanes)</p>'+
            '</div>';
        }}else{{
          document.getElementById('userLaneImpact').innerHTML='';
        }}

        return;
      }}

      const src=String(AREA_NODES[i]);
      const dist={{}},visited=new Set();
      dist[src]=0;
      let pq=[[0,src]];
      while(pq.length){{
        const[cd,cur]=pq.shift();
        if(visited.has(cur))continue;
        visited.add(cur);
        for(const{{n:nb,w}}of(adj[cur]||[])){{
          if(visited.has(nb))continue;
          const nd=cd+w;
          if(dist[nb]===undefined||nd<dist[nb]){{
            dist[nb]=nd;
            let ins=pq.findIndex(x=>x[0]>nd);
            if(ins<0)ins=pq.length;
            pq.splice(ins,0,[nd,nb]);
          }}
        }}
      }}

      for(let j=0;j<n;j++){{
        if(i===j)continue;
        const dstNode=String(AREA_NODES[j]);
        if(dist[dstNode]!==undefined){{
          const tau=Math.max(dist[dstNode]/1000,0.1);
          const decay=Math.pow(tau,theta);
          acc_orig[i]+=destWeight(j)*decay;
          acc_dest[j]+=AREA_POP[i]*decay;
          totalN+=AREA_POP[i]*destWeight(j)*decay;
        }}
      }}

      processed++;
      if(processed%10===0){{
        prog.innerHTML="<p>Processing areas: "+processed+"/"+n+" ("+Math.round(100*processed/n)+"%)</p>";
      }}
      setTimeout(()=>processArea(i+1),0);
    }}

    processArea(0);
  }},50);
}};

// Initial render
buildLaneList();
buildUserLaneList();
updateComputePanel();
// No longer compute baseline on startup - will compute when "Compute Accessibility" is clicked
</script>

<!-- Methodology Modal -->
<div id="methodModal" onclick="if(event.target===this)this.style.display='none'" style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.7);z-index:9999;justify-content:center;align-items:center">
  <div style="background:#fff;max-width:900px;max-height:90vh;overflow-y:auto;padding:30px;border-radius:8px;position:relative;margin:20px">
    <button onclick="document.getElementById('methodModal').style.display='none'" style="position:absolute;top:10px;right:15px;background:none;border:none;font-size:24px;cursor:pointer">&times;</button>
    <h1 style="color:#2c3e50;margin-top:0">Jerusalem Bike Lane Analysis - Methodology</h1>

    <h2 style="color:#34495e">Overview</h2>
    <p>This tool ranks proposed ("wishing list") bike lanes by their potential contribution to city-wide accessibility. It uses a gravity-based accessibility model to measure how well people can reach destinations across the city, with bike lanes significantly reducing the effective travel cost.</p>

    <h2 style="color:#34495e">The Accessibility Model</h2>
    <h3>Core Formula</h3>
    <p style="background:#f5f5f5;padding:15px;border-radius:4px;font-family:monospace;font-size:1.1em">
      N = &Sigma;<sub>i</sub> &Sigma;<sub>j</sub> P<sub>i</sub> &times; D<sub>j</sub> &times; &tau;<sub>ij</sub><sup>&theta;</sup>
    </p>
    <p>Where:</p>
    <ul>
      <li><b>P<sub>i</sub></b> = Population of area i (potential trip origins)</li>
      <li><b>D<sub>j</sub></b> = Weighted destination attractiveness of area j</li>
      <li><b>&tau;<sub>ij</sub></b> = Travel cost (shortest path distance in km) from area i to area j</li>
      <li><b>&theta;</b> = Distance decay parameter (negative, typically -1 to -2)</li>
    </ul>

    <h3>Multi-Destination Model</h3>
    <p>The destination weight D<sub>j</sub> combines multiple destination types:</p>
    <p style="background:#f5f5f5;padding:15px;border-radius:4px;font-family:monospace;font-size:1.1em">
      D<sub>j</sub> = w<sub>emp</sub> &times; E<sub>j</sub> + w<sub>edu</sub> &times; S<sub>j</sub> + w<sub>transit</sub> &times; T<sub>j</sub>
    </p>
    <ul>
      <li><b>E<sub>j</sub></b> = Employment in area j</li>
      <li><b>S<sub>j</sub></b> = Students in academic institutions in area j</li>
      <li><b>T<sub>j</sub></b> = Daily transit passengers in area j (train stations)</li>
      <li><b>w<sub>emp</sub>, w<sub>edu</sub>, w<sub>transit</sub></b> = User-adjustable weights (default: equal)</li>
    </ul>
    <p>Institutions include Hebrew University (3 campuses), Bezalel, Hadassah College, Machon Lev, David Yellin, Azrieli, Music Academy, Sam Spiegel, Musrara, and Al-Quds University (~53,600 students total). Transit includes Yitzhak Navon station (~15,000 daily passengers).</p>

    <h3>Parameters</h3>
    <h4>K - No-Lane Penalty</h4>
    <p>Roads without bike lanes are penalized by multiplying their length by K and a road discomfort factor:</p>
    <ul>
      <li><code>weight = length</code> for roads WITH bike lanes</li>
      <li><code>weight = length &times; K &times; discomfort</code> for roads WITHOUT bike lanes</li>
    </ul>
    <p>Higher K values mean cyclists strongly prefer bike lanes: K=10 (mild), K=100 (strong, default), K=500 (very strong).</p>

    <h4>Road Discomfort Factor</h4>
    <p>Roads are classified by OSM highway type. Major roads receive higher discomfort penalties:</p>
    <table style="width:100%;border-collapse:collapse;margin-bottom:15px;font-size:0.9em">
      <tr style="background:#34495e;color:#fff">
        <th style="padding:6px;border:1px solid #ddd">Road Type</th>
        <th style="padding:6px;border:1px solid #ddd">Factor</th>
        <th style="padding:6px;border:1px solid #ddd">Examples</th>
      </tr>
      <tr><td style="padding:6px;border:1px solid #ddd">residential</td><td style="padding:6px;border:1px solid #ddd">1.0</td><td style="padding:6px;border:1px solid #ddd">Neighborhood streets</td></tr>
      <tr style="background:#f9f9f9"><td style="padding:6px;border:1px solid #ddd">tertiary</td><td style="padding:6px;border:1px solid #ddd">1.5</td><td style="padding:6px;border:1px solid #ddd">Minor urban roads</td></tr>
      <tr><td style="padding:6px;border:1px solid #ddd">secondary</td><td style="padding:6px;border:1px solid #ddd">2.0</td><td style="padding:6px;border:1px solid #ddd">Urban arterials</td></tr>
      <tr style="background:#f9f9f9"><td style="padding:6px;border:1px solid #ddd">primary</td><td style="padding:6px;border:1px solid #ddd">3.0</td><td style="padding:6px;border:1px solid #ddd">Golda Meir, Herzl</td></tr>
      <tr><td style="padding:6px;border:1px solid #ddd">trunk</td><td style="padding:6px;border:1px solid #ddd">4.0</td><td style="padding:6px;border:1px solid #ddd">Bazak, Derech Hebron</td></tr>
      <tr style="background:#f9f9f9"><td style="padding:6px;border:1px solid #ddd">motorway</td><td style="padding:6px;border:1px solid #ddd">5.0</td><td style="padding:6px;border:1px solid #ddd">Begin Expressway</td></tr>
    </table>

    <h4>&theta; (Theta) - Distance Decay</h4>
    <p>Controls how quickly accessibility decreases with distance:</p>
    <ul>
      <li>&theta; = -0.5: Slow decay (long trips acceptable)</li>
      <li>&theta; = -1.0: Moderate decay (default)</li>
      <li>&theta; = -2.0: Fast decay (only nearby destinations matter)</li>
    </ul>
    <p style="background:#f5f5f5;padding:10px;border-radius:4px;font-size:0.9em">
      <b>References:</b><br>
      Donaldson, D., &amp; Hornbeck, R. (2016). Railroads and American economic growth: A "market access" approach. <i>The Quarterly Journal of Economics</i>, 131(2), 799-858.<br>
      Tsivanidis, N. (2024). Evaluating the Impact of Urban Transit Infrastructure: Evidence from Bogotá's TransMilenio. <i>American Economic Review</i>, 116(2), 418-463.
    </p>

    <h2 style="color:#34495e">Data Sources</h2>
    <table style="width:100%;border-collapse:collapse;margin-bottom:20px;font-size:0.9em">
      <tr style="background:#34495e;color:#fff">
        <th style="padding:8px;text-align:left;border:1px solid #ddd">Data</th>
        <th style="padding:8px;text-align:left;border:1px solid #ddd">Source</th>
      </tr>
      <tr><td style="padding:8px;border:1px solid #ddd">Statistical Areas</td><td style="padding:8px;border:1px solid #ddd">Jerusalem Transportation Master Plan Team</td></tr>
      <tr style="background:#f9f9f9"><td style="padding:8px;border:1px solid #ddd">Population</td><td style="padding:8px;border:1px solid #ddd">Jerusalem Transportation Master Plan Team</td></tr>
      <tr><td style="padding:8px;border:1px solid #ddd">Employment</td><td style="padding:8px;border:1px solid #ddd">Jerusalem Transportation Master Plan Team</td></tr>
      <tr style="background:#f9f9f9"><td style="padding:8px;border:1px solid #ddd">Completed Bike Lanes</td><td style="padding:8px;border:1px solid #ddd">Jerusalem Transportation Master Plan Team</td></tr>
      <tr><td style="padding:8px;border:1px solid #ddd">Under Construction Bike Lanes</td><td style="padding:8px;border:1px solid #ddd">Jerusalem Transportation Master Plan Team</td></tr>
      <tr style="background:#f9f9f9"><td style="padding:8px;border:1px solid #ddd">Planned Bike Lanes</td><td style="padding:8px;border:1px solid #ddd">Jerusalem Transportation Master Plan Team</td></tr>
      <tr><td style="padding:8px;border:1px solid #ddd">Checked Bike Lanes</td><td style="padding:8px;border:1px solid #ddd">Jerusalem Transportation Master Plan Team</td></tr>
      <tr style="background:#f9f9f9"><td style="padding:8px;border:1px solid #ddd">Wishing List Bike Lanes</td><td style="padding:8px;border:1px solid #ddd">The author</td></tr>
      <tr><td style="padding:8px;border:1px solid #ddd">Road Network</td><td style="padding:8px;border:1px solid #ddd">OpenStreetMap</td></tr>
      <tr style="background:#f9f9f9"><td style="padding:8px;border:1px solid #ddd">Road Classification</td><td style="padding:8px;border:1px solid #ddd">OpenStreetMap (Overpass API)</td></tr>
      <tr><td style="padding:8px;border:1px solid #ddd">Academic Institutions</td><td style="padding:8px;border:1px solid #ddd">OpenStreetMap + student count estimates</td></tr>
      <tr style="background:#f9f9f9"><td style="padding:8px;border:1px solid #ddd">Train Station Passengers</td><td style="padding:8px;border:1px solid #ddd">Israel Railways estimates</td></tr>
    </table>

    <h2 style="color:#34495e">Road Network Construction</h2>
    <p>The road network is built from Jerusalem road data (from OpenStreetMap, KML format) with the following process:</p>
    <ol>
      <li><b>Node Creation</b>: Road endpoints are snapped to a grid (15m tolerance) to create a connected graph</li>
      <li><b>Edge Creation</b>: Each road segment becomes an edge with its physical length as the base weight</li>
      <li><b>Bike Lane Matching</b>: Existing bike lanes are spatially matched to road edges using a 15m buffer and 50% overlap threshold</li>
      <li><b>Coordinate Systems</b>: Calculations use Israeli TM (EPSG:2039) for accurate distance; display uses WGS84 (EPSG:4326)</li>
    </ol>

    <h2 style="color:#34495e">Network Connectivity</h2>
    <p>The tool ensures the network is fully connected through several mechanisms:</p>
    <h4>Node Merging</h4>
    <ul>
      <li><b>Tolerance</b>: Nodes within 15 meters are merged into a single node</li>
      <li><b>Purpose</b>: Handles imprecise GPS coordinates and ensures lane endpoints connect properly to roads</li>
    </ul>
    <h4>Intersection Detection</h4>
    <ul>
      <li>Bike lanes are overlaid on the road network</li>
      <li>Intersection points between bike lanes and roads are detected automatically</li>
      <li>New nodes are created at every intersection point</li>
      <li>Edges are split at intersection points to enable routing through the network</li>
    </ul>
    <h4>Gap Connection Algorithm</h4>
    <p>For bike lanes with gaps between segments:</p>
    <ol>
      <li><b>Endpoint Extraction</b>: Extract start and end points from all lane geometries</li>
      <li><b>KD-Tree Indexing</b>: Build spatial index for efficient nearest-neighbor queries</li>
      <li><b>Dangling Endpoint Detection</b>: Identify endpoints not touching other lanes (within 1m tolerance)</li>
      <li><b>Gap Bridging</b>: Connect dangling endpoints to nearest neighbor within 50m tolerance</li>
    </ol>
    <h4>Component Connection</h4>
    <p>For disconnected network components:</p>
    <ol>
      <li><b>Component Detection</b>: Find all connected components using graph algorithms</li>
      <li><b>Minimum Spanning Tree Approach</b>: Connect isolated components by adding edges between closest nodes</li>
    </ol>
    <p style="background:#f5f5f5;padding:10px;border-radius:4px;font-size:0.9em">This connectivity fixing is essential because raw GIS data often has small gaps, coordinate mismatches, or isolated segments that would otherwise break shortest path calculations.</p>

    <h2 style="color:#34495e">Shortest Path Algorithm</h2>
    <p>We use <b>Dijkstra's algorithm</b> to compute shortest paths between all area centroids:</p>
    <ul>
      <li><b>Graph</b>: Undirected weighted graph where edge weight = length &times; K &times; discomfort for roads without bike lanes</li>
      <li><b>Source</b>: Nearest network node to each area centroid</li>
      <li><b>Output</b>: Distance matrix &tau;<sub>ij</sub> between all area pairs</li>
    </ul>
    <p style="background:#f5f5f5;padding:10px;border-radius:4px;font-size:0.9em">
      <b>Reference:</b> Dijkstra, E.W. (1959). A note on two problems in connexion with graphs. <i>Numerische Mathematik</i>, 1(1), 269-271.
    </p>

    <h2 style="color:#34495e">Online Computation</h2>
    <p>All accessibility calculations are performed in the browser using JavaScript:</p>
    <ol>
      <li><b>Baseline Computation</b>: When K or &theta; changes, compute accessibility with existing lanes only</li>
      <li><b>Network Update</b>: When wishing lanes are selected, mark their corresponding road edges as bike lanes (weight = length instead of length &times; K &times; discomfort)</li>
      <li><b>Full Recomputation</b>: Run Dijkstra from each area centroid to compute new &tau;<sub>ij</sub> matrix</li>
      <li><b>Accessibility Aggregation</b>: Sum P<sub>i</sub> &times; D<sub>j</sub> &times; &tau;<sub>ij</sub><sup>&theta;</sup> for all pairs</li>
    </ol>

    <h2 style="color:#34495e">How Lanes Are Ranked</h2>
    <ol>
      <li><b>Baseline Calculation</b>: Compute total N using existing bike lanes only</li>
      <li><b>With Selected Lanes</b>: Add selected wishing lanes and recompute N</li>
      <li><b>Improvement</b>: %&Delta; = 100 &times; (N<sub>new</sub> - N<sub>baseline</sub>) / N<sub>baseline</sub></li>
    </ol>

    <h2 style="color:#34495e">Area Accessibility</h2>
    <p><b>Origin Accessibility</b> (destinations reachable from area i):</p>
    <p style="background:#f5f5f5;padding:10px;border-radius:4px;font-family:monospace">
      acc<sub>origin</sub>[i] = &Sigma;<sub>j</sub> D<sub>j</sub> &times; &tau;<sub>ij</sub><sup>&theta;</sup>
    </p>
    <p>Measures how many weighted destinations (jobs, students, transit) area i residents can access.</p>

    <p><b>Destination Accessibility</b> (people who can reach area j):</p>
    <p style="background:#f5f5f5;padding:10px;border-radius:4px;font-family:monospace">
      acc<sub>dest</sub>[j] = &Sigma;<sub>i</sub> P<sub>i</sub> &times; &tau;<sub>ij</sub><sup>&theta;</sup>
    </p>
    <p>Measures how many people can reach area j (e.g., a workplace or campus).</p>

    <h2 style="color:#34495e">Visualization</h2>
    <ul>
      <li><b>Area Colors</b>: Spectral colormap (blue &rarr; cyan &rarr; yellow &rarr; orange &rarr; red) with logarithmic scaling</li>
      <li><b>Accessibility Mode</b>: Shows absolute accessibility values</li>
      <li><b>Change Mode</b>: Shows percentage improvement from baseline after adding selected lanes</li>
    </ul>

    <div style="text-align:center;margin-top:20px">
      <button onclick="document.getElementById('methodModal').style.display='none'" style="padding:10px 30px;background:#27ae60;color:#fff;border:none;border-radius:4px;font-size:14px;cursor:pointer">Close</button>
    </div>
  </div>
</div>
<button class="sidebar-toggle" onclick="toggleSidebar()" id="sidebarToggle">&#9776;</button>
<script>
function toggleSidebar(){{
  const sb=document.querySelector('.sidebar');
  const btn=document.getElementById('sidebarToggle');
  sb.classList.toggle('mobile-hidden');
  btn.innerHTML=sb.classList.contains('mobile-hidden')?'&#9776;':'&times;';
  // Invalidate map size when sidebar toggled
  setTimeout(()=>{{if(typeof map!=='undefined')map.invalidateSize();}},100);
}}
function togglePanelSection(cls,btnId,label){{
  const el=document.querySelector('.'+cls);
  const btn=document.getElementById(btnId);
  const collapsed=el.classList.toggle('collapsed');
  btn.innerHTML=label+' '+(collapsed?'&#9654;':'&#9660;');
  setTimeout(()=>{{if(typeof map!=='undefined')map.invalidateSize();}},50);
}}
let _savedSidebarSize=null;
function toggleSidebarDesktop(){{
  const sb=document.querySelector('.sidebar');
  const btn=document.getElementById('btnSidebar');
  sb.classList.remove('mobile-hidden');
  const isMobile=window.innerWidth<=768;
  if(!sb.classList.contains('collapsed')){{
    // Save current size (inline or computed), then clear inline so CSS class works
    _savedSidebarSize=isMobile?sb.offsetHeight+'px':sb.offsetWidth+'px';
    sb.style.width='';sb.style.height='';sb.style.overflowY='';
    sb.classList.add('collapsed');
    btn.innerHTML='Panel &#9654;';
  }}else{{
    sb.classList.remove('collapsed');
    if(_savedSidebarSize){{
      if(isMobile)sb.style.height=_savedSidebarSize;
      else sb.style.width=_savedSidebarSize;
    }}
    sb.style.overflowY='auto';
    btn.innerHTML='Panel &#9660;';
  }}
  setTimeout(()=>{{if(typeof map!=='undefined')map.invalidateSize();}},250);
}}
function toggleLegend(){{
  const content=document.getElementById('legendContent');
  const btn=document.getElementById('legendToggle');
  const collapsed=content.classList.toggle('collapsed');
  btn.innerHTML=collapsed?'&#9654;':'&#9660;';
}}
// Resize handle
(function(){{
  const handle=document.getElementById('resizeHandle');
  if(!handle)return;
  let dragging=false,startX,startY,startW,startH;
  function onMove(e){{
    if(!dragging)return;
    const sb=document.querySelector('.sidebar');
    const isMobile=window.innerWidth<=768;
    if(isMobile){{
      const dy=startY-(e.touches?e.touches[0].clientY:e.clientY);
      sb.style.height=Math.max(80,Math.min(window.innerHeight*0.8,startH+dy))+'px';
    }}else{{
      const dx=startX-(e.touches?e.touches[0].clientX:e.clientX);
      sb.style.width=Math.max(200,Math.min(window.innerWidth*0.6,startW+dx))+'px';
    }}
    sb.style.overflowY='auto';
    if(typeof map!=='undefined')map.invalidateSize();
  }}
  function onUp(){{
    dragging=false;
    handle.classList.remove('dragging');
    document.removeEventListener('mousemove',onMove);
    document.removeEventListener('mouseup',onUp);
    document.removeEventListener('touchmove',onMove);
    document.removeEventListener('touchend',onUp);
  }}
  function onDown(e){{
    if(document.querySelector('.sidebar').classList.contains('collapsed'))return;
    dragging=true;
    handle.classList.add('dragging');
    const sb=document.querySelector('.sidebar');
    startX=e.touches?e.touches[0].clientX:e.clientX;
    startY=e.touches?e.touches[0].clientY:e.clientY;
    startW=sb.offsetWidth;
    startH=sb.offsetHeight;
    document.addEventListener('mousemove',onMove);
    document.addEventListener('mouseup',onUp);
    document.addEventListener('touchmove',onMove,{{passive:false}});
    document.addEventListener('touchend',onUp);
    e.preventDefault();
  }}
  handle.addEventListener('mousedown',onDown);
  handle.addEventListener('touchstart',onDown,{{passive:false}});
}})();
// On mobile: auto-collapse controls to give more map space
if(window.innerWidth<=768){{
  const ctrl=document.querySelector('.controls');
  const btn=document.getElementById('btnParams');
  if(ctrl&&btn){{ctrl.classList.add('collapsed');btn.innerHTML='Params &#9654;';}}
}}
</script>
</body>
</html>'''


if __name__ == '__main__':
    main()
