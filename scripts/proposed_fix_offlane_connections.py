#!/usr/bin/env python3
"""
Proposed fix: Connect off-road bike lanes to the road network.

For bike lanes that are far from the road network:
1. Create nodes at bike lane endpoints
2. Connect those nodes to the nearest road network nodes
3. Create edges along the bike lane itself

This ensures every bike lane is reachable, even if it's in a park or far from roads.
"""
import sys
import os

import geopandas as gpd
import numpy as np
import networkx as nx
from pathlib import Path
from scipy.spatial import cKDTree
from shapely.geometry import Point, LineString
from shapely.strtree import STRtree
from shapely.ops import substring
from pyproj import Transformer
import fiona
import warnings

warnings.filterwarnings('ignore')

script_dir = Path(__file__).parent
data_dir = script_dir.parent  # Input files and main script are in parent directory
sys.path.insert(0, str(data_dir))  # Add parent to path for imports
fiona.drvsupport.supported_drivers['KML'] = 'rw'
TARGET_CRS = 2039
WGS84 = 4326
NODE_TOLERANCE = 15

def build_network_with_offroad_connections(roads_proj, bike_lanes_list, areas_proj=None,
                                            tolerance=NODE_TOLERANCE,
                                            virtual_threshold=50,
                                            connector_threshold=1000):
    """
    Build network with improved off-road bike lane connections.

    New parameter:
    - connector_threshold: Max distance to create connector from bike lane to road network.
                          Any bike lane endpoint further than virtual_threshold but within
                          connector_threshold will get a connector edge to nearest road node.
    """
    from shapely.strtree import STRtree

    G = nx.Graph()
    coord_to_node = {}
    node_coords = {}
    node_counter = [0]
    edge_to_geom = {}

    def get_or_create_node(x, y):
        key = (round(x / tolerance) * tolerance, round(y / tolerance) * tolerance)
        if key in coord_to_node:
            return coord_to_node[key]
        nid = node_counter[0]
        node_counter[0] += 1
        coord_to_node[key] = nid
        node_coords[nid] = (x, y)
        return nid

    # Add road segments
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
                G.add_edge(s, e, length=geom.length, has_bike_lane=False)
                edge_key = (min(s, e), max(s, e))
                edge_to_geom[edge_key] = geom
                road_geoms.append(geom)
                road_edges.append(edge_key)

    road_tree_spatial = STRtree(road_geoms) if road_geoms else None
    BUFFER_DIST = 15

    # Mark bike lane roads (same as before)
    def mark_bike_lane_roads(line_geom):
        if road_tree_spatial is None:
            return
        buffered = line_geom.buffer(BUFFER_DIST)
        candidate_indices = road_tree_spatial.query(buffered)
        for idx in candidate_indices:
            road_geom = road_geoms[idx]
            try:
                intersection = road_geom.intersection(buffered)
                if intersection.is_empty:
                    continue
                overlap_ratio = intersection.length / road_geom.length if road_geom.length > 0 else 0
                should_mark = (
                    overlap_ratio > 0.5 or
                    (road_geom.length < 50 and overlap_ratio > 0.3) or
                    intersection.length > 20
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
            if geom.geom_type == 'LineString':
                mark_bike_lane_roads(geom)
            elif geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    mark_bike_lane_roads(line)

    # === NEW: Create off-road bike lane connections ===
    # For lanes that are far from roads, create:
    # 1. Nodes at lane endpoints
    # 2. Connector edges to nearest road nodes
    # 3. Edges along the lane itself

    print("\n  Creating off-road bike lane connections...")
    offroad_connections = 0
    offroad_lanes_connected = 0

    def connect_offroad_lane(line_geom):
        """Connect an off-road bike lane to the road network."""
        nonlocal offroad_connections, offroad_lanes_connected

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

        # If both endpoints are already within virtual_threshold, standard logic handles it
        if start_dist <= virtual_threshold and end_dist <= virtual_threshold:
            return 0

        connections_made = 0

        # Create nodes at lane endpoints if they're far from existing nodes
        if start_dist > virtual_threshold:
            start_node = get_or_create_node(start.x, start.y)
            if start_dist <= connector_threshold:
                # Connect to nearest road node
                nearest_road_node = current_node_ids[start_idx]
                if start_node != nearest_road_node and not G.has_edge(start_node, nearest_road_node):
                    # This is a connector edge - NOT a bike lane, but allows access
                    G.add_edge(start_node, nearest_road_node,
                              length=start_dist,
                              has_bike_lane=False,  # Walking/access connector
                              is_connector=True)
                    connections_made += 1

        if end_dist > virtual_threshold:
            end_node = get_or_create_node(end.x, end.y)
            # Re-query since we may have added start_node
            current_node_ids = list(node_coords.keys())
            current_coords = np.array([node_coords[n] for n in current_node_ids])
            current_tree = cKDTree(current_coords)
            end_dist, end_idx = current_tree.query([end.x, end.y])

            if end_dist <= connector_threshold:
                nearest_road_node = current_node_ids[end_idx]
                if end_node != nearest_road_node and not G.has_edge(end_node, nearest_road_node):
                    G.add_edge(end_node, nearest_road_node,
                              length=end_dist,
                              has_bike_lane=False,
                              is_connector=True)
                    connections_made += 1

        # Now create edges along the bike lane itself
        # Create intermediate nodes at regular intervals
        SAMPLE_INTERVAL = 100  # Create node every 100m

        if line_geom.length > SAMPLE_INTERVAL:
            num_segments = max(2, int(line_geom.length / SAMPLE_INTERVAL))
            prev_node = None

            for i in range(num_segments + 1):
                t = i / num_segments
                pt = line_geom.interpolate(t, normalized=True)
                node = get_or_create_node(pt.x, pt.y)

                if prev_node is not None and prev_node != node:
                    # Calculate actual distance along lane
                    prev_t = (i - 1) / num_segments
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

        if connections_made > 0:
            offroad_connections += connections_made
            offroad_lanes_connected += 1

        return connections_made

    # Process all bike lanes for off-road connections
    for bl_gdf in bike_lanes_list:
        if bl_gdf is None or len(bl_gdf) == 0:
            continue
        bl_proj = bl_gdf.to_crs(TARGET_CRS)
        for _, row in bl_proj.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == 'LineString':
                connect_offroad_lane(geom)
            elif geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    connect_offroad_lane(line)

    print(f"  Off-road lanes connected: {offroad_lanes_connected}")
    print(f"  Off-road connections created: {offroad_connections}")

    # Standard virtual edge creation (for lanes near roads)
    VIRTUAL_EDGE_THRESHOLD = virtual_threshold

    def create_virtual_edges_for_lane(line_geom):
        if len(node_coords) == 0:
            return 0

        current_node_ids = list(node_coords.keys())
        current_coords = np.array([node_coords[n] for n in current_node_ids])
        if len(current_coords) < 2:
            return 0
        current_tree = cKDTree(current_coords)

        line_length = line_geom.length
        if line_length < 10:
            return 0

        nearby_nodes = []
        line_buffer = line_geom.buffer(VIRTUAL_EDGE_THRESHOLD)

        for idx, nid in enumerate(current_node_ids):
            pt = Point(current_coords[idx])
            if line_buffer.contains(pt):
                proj_dist = line_geom.project(pt)
                perp_dist = pt.distance(line_geom)
                if perp_dist <= VIRTUAL_EDGE_THRESHOLD:
                    nearby_nodes.append({
                        'node_id': nid,
                        'proj_dist': proj_dist,
                        'perp_dist': perp_dist
                    })

        nearby_nodes.sort(key=lambda x: x['proj_dist'])

        virtual_edge_count = 0
        for i in range(len(nearby_nodes) - 1):
            n1 = nearby_nodes[i]
            n2 = nearby_nodes[i + 1]
            edge_len = n2['proj_dist'] - n1['proj_dist']

            if edge_len > 5:
                node_a, node_b = n1['node_id'], n2['node_id']
                if not G.has_edge(node_a, node_b):
                    G.add_edge(node_a, node_b, length=edge_len, has_bike_lane=True, is_virtual=True)
                    virtual_edge_count += 1
                    try:
                        edge_geom = substring(line_geom, n1['proj_dist'], n2['proj_dist'])
                        if edge_geom and not edge_geom.is_empty and edge_geom.geom_type == 'LineString':
                            edge_key = (min(node_a, node_b), max(node_a, node_b))
                            edge_to_geom[edge_key] = edge_geom
                    except:
                        pass
                elif not G[node_a][node_b].get('has_bike_lane'):
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
                total_virtual_edges += count
            elif geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    count = create_virtual_edges_for_lane(line)
                    total_virtual_edges += count

    print(f"  Standard virtual edges: {total_virtual_edges}")

    # Connect area centroids (same as before)
    if areas_proj is not None and len(road_geoms) > 0:
        road_tree_for_areas = STRtree(road_geoms)
        for _, area_row in areas_proj.iterrows():
            centroid = area_row.geometry.centroid
            centroid_pt = Point(centroid.x, centroid.y)
            nearest_idx = road_tree_for_areas.nearest(centroid_pt)
            nearest_road = road_geoms[nearest_idx]
            nearest_point_on_road = nearest_road.interpolate(nearest_road.project(centroid_pt))
            dist_to_road = centroid_pt.distance(nearest_point_on_road)
            road_node = get_or_create_node(nearest_point_on_road.x, nearest_point_on_road.y)

            if dist_to_road > tolerance:
                centroid_node = get_or_create_node(centroid.x, centroid.y)
                if centroid_node != road_node:
                    G.add_edge(centroid_node, road_node, length=dist_to_road, has_bike_lane=False)

    # Ensure connectivity
    node_ids = list(node_coords.keys())
    coords_array = np.array([node_coords[n] for n in node_ids])
    node_tree = cKDTree(coords_array) if len(coords_array) > 0 else None

    if areas_proj is not None and node_tree is not None:
        largest_cc = max(nx.connected_components(G), key=len)
        cc_nodes = [n for n in node_ids if n in largest_cc]
        if cc_nodes:
            cc_coords = np.array([node_coords[n] for n in cc_nodes])
            cc_tree = cKDTree(cc_coords)

            for _, area_row in areas_proj.iterrows():
                centroid = area_row.geometry.centroid
                _, nearest_idx = node_tree.query([centroid.x, centroid.y])
                nearest_node = node_ids[nearest_idx]

                if nearest_node not in largest_cc:
                    dist, cc_idx = cc_tree.query([centroid.x, centroid.y])
                    cc_node = cc_nodes[cc_idx]

                    if nearest_node != cc_node:
                        nearest_coord = node_coords[nearest_node]
                        cc_coord = node_coords[cc_node]
                        edge_length = np.sqrt((nearest_coord[0] - cc_coord[0])**2 +
                                             (nearest_coord[1] - cc_coord[1])**2)
                        G.add_edge(nearest_node, cc_node, length=edge_length, has_bike_lane=False)
                        largest_cc = max(nx.connected_components(G), key=len)

    return G, node_coords, node_tree, node_ids, edge_to_geom


def test_improved_network():
    """Test the improved network building."""
    print("=" * 70)
    print("TESTING IMPROVED OFF-ROAD BIKE LANE CONNECTIONS")
    print("=" * 70)

    # Load data
    from generate_interactive_map import load_data

    areas, roads, completed, construction, plan, check, wishing = load_data()
    areas_proj = areas.to_crs(TARGET_CRS)
    roads_proj = roads.to_crs(TARGET_CRS)

    print("\n1. Building ORIGINAL network...")
    from generate_interactive_map import build_network as original_build
    G_orig, nc_orig, nt_orig, ni_orig, _ = original_build(
        roads_proj, [completed, construction, plan, check, wishing], areas_proj
    )
    print(f"   Nodes: {G_orig.number_of_nodes()}, Edges: {G_orig.number_of_edges()}")

    print("\n2. Building IMPROVED network...")
    G_new, nc_new, nt_new, ni_new, _ = build_network_with_offroad_connections(
        roads_proj, [completed, construction, plan, check, wishing], areas_proj,
        connector_threshold=1000  # Connect lanes up to 1km from roads
    )
    print(f"   Nodes: {G_new.number_of_nodes()}, Edges: {G_new.number_of_edges()}")

    # Compare bike lane coverage
    print("\n3. Comparing bike lane coverage...")

    orig_bike_edges = sum(1 for u, v, d in G_orig.edges(data=True) if d.get('has_bike_lane'))
    new_bike_edges = sum(1 for u, v, d in G_new.edges(data=True) if d.get('has_bike_lane'))

    print(f"   Original bike edges: {orig_bike_edges}")
    print(f"   New bike edges: {new_bike_edges}")
    print(f"   Improvement: +{new_bike_edges - orig_bike_edges} edges")

    # Test specific problem locations
    print("\n4. Testing problem locations...")
    transformer_inv = Transformer.from_crs(WGS84, TARGET_CRS, always_xy=True)

    test_locations = [
        ("Western area (was 656m)", 31.78065, 35.15645),
        ("Southwest (was 259m)", 31.75141, 35.15929),
        ("Park trail (was 49m)", 31.74313, 35.18105),
    ]

    for name, lat, lon in test_locations:
        x, y = transformer_inv.transform(lon, lat)

        # Check original network
        orig_coords = np.array([nc_orig[n] for n in ni_orig])
        orig_tree = cKDTree(orig_coords)
        orig_dist, _ = orig_tree.query([x, y])

        # Check new network
        new_coords = np.array([nc_new[n] for n in ni_new])
        new_tree = cKDTree(new_coords)
        new_dist, _ = new_tree.query([x, y])

        print(f"\n   {name}:")
        print(f"     Original: {orig_dist:.0f}m to nearest node")
        print(f"     Improved: {new_dist:.0f}m to nearest node")

    # Count connector edges
    connector_edges = sum(1 for u, v, d in G_new.edges(data=True) if d.get('is_connector'))
    print(f"\n5. Connector edges created: {connector_edges}")

    return G_new, nc_new


if __name__ == "__main__":
    G, node_coords = test_improved_network()
