#!/usr/bin/env python3
"""
Identify specific gaps in bike lane connectivity and find missing connections.
"""
import geopandas as gpd
import numpy as np
import networkx as nx
from pathlib import Path
from scipy.spatial import cKDTree
from shapely.geometry import Point, LineString
from pyproj import Transformer
import fiona
import warnings
import json

warnings.filterwarnings('ignore')

script_dir = Path(__file__).parent
fiona.drvsupport.supported_drivers['KML'] = 'rw'
TARGET_CRS = 2039
WGS84 = 4326
NODE_TOLERANCE = 15
VIRTUAL_EDGE_THRESHOLD = 50

def load_all_data():
    """Load all data files."""
    areas = gpd.read_file(script_dir / "jer_areas.shp")
    areas = areas[areas['in_jeru'] == 1].copy()
    roads = gpd.read_file(script_dir / "jerusalem_roads.kml", driver='KML')

    layers = {}
    layer_files = {
        'completed': 'bike_lanes_completed.kml',
        'construction': 'bike_lanes_construction.kml',
        'plan': 'bike_lanes_plan.kml',
        'check': 'bike_lanes_check.kml',
        'wishing': 'bike_lanes_wishing_list.kml'
    }

    for name, filename in layer_files.items():
        try:
            gdf = gpd.read_file(script_dir / filename, driver='KML', on_invalid='ignore')
            gdf = gdf[gdf.geometry.notnull()].copy()
            layers[name] = gdf
        except:
            layers[name] = gpd.GeoDataFrame(columns=['geometry', 'Name'], geometry='geometry', crs='EPSG:4326')

    return areas, roads, layers


def analyze_bike_lane_features():
    """Analyze individual bike lane features for connectivity."""
    print("=" * 70)
    print("BIKE LANE FEATURE ANALYSIS")
    print("=" * 70)

    areas, roads, layers = load_all_data()
    roads_proj = roads.to_crs(TARGET_CRS)

    # Build road node network
    print("\n1. Building road network nodes...")
    coord_to_node = {}
    node_coords = {}
    node_counter = [0]

    def get_or_create_node(x, y):
        key = (round(x / NODE_TOLERANCE) * NODE_TOLERANCE, round(y / NODE_TOLERANCE) * NODE_TOLERANCE)
        if key in coord_to_node:
            return coord_to_node[key]
        nid = node_counter[0]
        node_counter[0] += 1
        coord_to_node[key] = nid
        node_coords[nid] = (x, y)
        return nid

    for _, row in roads_proj.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or geom.geom_type != 'LineString':
            continue
        coords = list(geom.coords)
        if len(coords) >= 2:
            get_or_create_node(coords[0][0], coords[0][1])
            get_or_create_node(coords[-1][0], coords[-1][1])

    print(f"   Road network has {len(node_coords)} nodes")

    # Build KD-tree for road nodes
    node_ids = list(node_coords.keys())
    node_coords_arr = np.array([node_coords[n] for n in node_ids])
    road_tree = cKDTree(node_coords_arr)

    # Analyze each bike lane layer
    print("\n2. Analyzing bike lane features...")
    transformer = Transformer.from_crs(TARGET_CRS, WGS84, always_xy=True)

    all_lane_issues = []

    for layer_name, gdf in layers.items():
        if len(gdf) == 0:
            continue

        gdf_proj = gdf.to_crs(TARGET_CRS)
        print(f"\n   === {layer_name.upper()} ({len(gdf)} features) ===")

        for idx, row in gdf_proj.iterrows():
            geom = row.geometry
            lane_name = row.get('Name', f'{layer_name}_{idx}')

            if geom is None or geom.is_empty:
                continue

            # Get all line geometries
            lines = []
            if geom.geom_type == 'LineString':
                lines = [geom]
            elif geom.geom_type == 'MultiLineString':
                lines = list(geom.geoms)

            for line in lines:
                if line.length < 10:
                    continue

                # Check endpoints
                start = Point(line.coords[0])
                end = Point(line.coords[-1])

                # Find nearest road nodes to endpoints
                start_dist, start_idx = road_tree.query([start.x, start.y])
                end_dist, end_idx = road_tree.query([end.x, end.y])

                # Count nodes within virtual edge threshold along the line
                line_buffer = line.buffer(VIRTUAL_EDGE_THRESHOLD)
                nearby_count = 0
                for i, nid in enumerate(node_ids):
                    pt = Point(node_coords_arr[i])
                    if line_buffer.contains(pt):
                        nearby_count += 1

                issues = []

                # Check if endpoints are far from road network
                if start_dist > VIRTUAL_EDGE_THRESHOLD:
                    issues.append(f"START disconnected ({start_dist:.0f}m from nearest road)")
                if end_dist > VIRTUAL_EDGE_THRESHOLD:
                    issues.append(f"END disconnected ({end_dist:.0f}m from nearest road)")

                # Check if lane has few road network connections
                if nearby_count < 2:
                    issues.append(f"Only {nearby_count} road nodes nearby (isolated)")

                if issues:
                    # Get center point for location
                    center = line.interpolate(0.5, normalized=True)
                    lon, lat = transformer.transform(center.x, center.y)

                    all_lane_issues.append({
                        'layer': layer_name,
                        'name': lane_name,
                        'length': line.length,
                        'issues': issues,
                        'lat': lat,
                        'lon': lon,
                        'nearby_nodes': nearby_count,
                        'start_dist': start_dist,
                        'end_dist': end_dist
                    })

    # Summary
    print("\n" + "=" * 70)
    print("LANES WITH CONNECTIVITY ISSUES")
    print("=" * 70)

    if all_lane_issues:
        # Group by layer
        for layer_name in layers.keys():
            layer_issues = [i for i in all_lane_issues if i['layer'] == layer_name]
            if layer_issues:
                print(f"\n{layer_name.upper()}:")
                for issue in layer_issues:
                    print(f"\n  {issue['name']} ({issue['length']:.0f}m)")
                    print(f"    Location: {issue['lat']:.5f}, {issue['lon']:.5f}")
                    print(f"    Nearby road nodes: {issue['nearby_nodes']}")
                    for i in issue['issues']:
                        print(f"    - {i}")
    else:
        print("\nNo major connectivity issues found!")

    return all_lane_issues


def find_bike_network_gaps():
    """Find gaps between bike lane network components."""
    print("\n" + "=" * 70)
    print("BIKE NETWORK GAP ANALYSIS")
    print("=" * 70)

    areas, roads, layers = load_all_data()
    roads_proj = roads.to_crs(TARGET_CRS)

    from shapely.strtree import STRtree

    # Build full network with bike lane marking
    G = nx.Graph()
    coord_to_node = {}
    node_coords = {}
    node_counter = [0]

    def get_or_create_node(x, y):
        key = (round(x / NODE_TOLERANCE) * NODE_TOLERANCE, round(y / NODE_TOLERANCE) * NODE_TOLERANCE)
        if key in coord_to_node:
            return coord_to_node[key]
        nid = node_counter[0]
        node_counter[0] += 1
        coord_to_node[key] = nid
        node_coords[nid] = (x, y)
        return nid

    # Add roads
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
                road_geoms.append(geom)
                road_edges.append(edge_key)

    road_tree = STRtree(road_geoms) if road_geoms else None
    BUFFER_DIST = 15

    # Mark bike lane roads
    def mark_roads(gdf):
        if len(gdf) == 0 or road_tree is None:
            return
        gdf_proj = gdf.to_crs(TARGET_CRS)
        for _, row in gdf_proj.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms)
            for line in lines:
                buffered = line.buffer(BUFFER_DIST)
                for idx in road_tree.query(buffered):
                    road_geom = road_geoms[idx]
                    try:
                        intersection = road_geom.intersection(buffered)
                        if intersection.is_empty:
                            continue
                        overlap = intersection.length / road_geom.length if road_geom.length > 0 else 0
                        if overlap > 0.5 or (road_geom.length < 50 and overlap > 0.3) or intersection.length > 20:
                            s, e = road_edges[idx]
                            if G.has_edge(s, e):
                                G[s][e]['has_bike_lane'] = True
                    except:
                        pass

    # Create virtual edges
    def create_virtuals(gdf):
        virtual_count = 0
        if len(gdf) == 0:
            return 0
        gdf_proj = gdf.to_crs(TARGET_CRS)
        node_ids = list(node_coords.keys())
        coords_arr = np.array([node_coords[n] for n in node_ids])
        tree = cKDTree(coords_arr)

        for _, row in gdf_proj.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms)
            for line in lines:
                if line.length < 10:
                    continue
                line_buffer = line.buffer(VIRTUAL_EDGE_THRESHOLD)
                nearby = []
                for i, nid in enumerate(node_ids):
                    pt = Point(coords_arr[i])
                    if line_buffer.contains(pt):
                        proj = line.project(pt)
                        dist = pt.distance(line)
                        if dist <= VIRTUAL_EDGE_THRESHOLD:
                            nearby.append({'nid': nid, 'proj': proj, 'dist': dist})

                nearby.sort(key=lambda x: x['proj'])
                for i in range(len(nearby) - 1):
                    n1, n2 = nearby[i]['nid'], nearby[i+1]['nid']
                    edge_len = nearby[i+1]['proj'] - nearby[i]['proj']
                    if edge_len > 5 and not G.has_edge(n1, n2):
                        G.add_edge(n1, n2, length=edge_len, has_bike_lane=True, is_virtual=True)
                        virtual_count += 1

        return virtual_count

    print("\n1. Building network with bike lanes...")
    for name, gdf in layers.items():
        mark_roads(gdf)
        v = create_virtuals(gdf)
        print(f"   {name}: {v} virtual edges")

    # Extract bike-only network
    bike_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('has_bike_lane')]
    G_bike = G.edge_subgraph(bike_edges).copy()

    print(f"\n2. Bike network: {G_bike.number_of_nodes()} nodes, {G_bike.number_of_edges()} edges")

    # Find connected components
    components = list(nx.connected_components(G_bike))
    print(f"   Number of components: {len(components)}")

    if len(components) <= 1:
        print("   Bike network is fully connected!")
        return

    # Sort by size
    components = sorted(components, key=len, reverse=True)

    print(f"\n3. Component sizes:")
    for i, cc in enumerate(components[:15]):
        print(f"   Component {i+1}: {len(cc)} nodes")

    # Find nearest connections between components
    print("\n4. Potential connections between components...")
    transformer = Transformer.from_crs(TARGET_CRS, WGS84, always_xy=True)

    # Get coordinates for each component
    def get_component_coords(cc):
        return np.array([node_coords[n] for n in cc])

    # Find gaps between top components
    top_components = components[:10]  # Top 10 largest
    gaps = []

    for i, cc1 in enumerate(top_components):
        cc1_nodes = list(cc1)
        cc1_coords = get_component_coords(cc1)
        cc1_tree = cKDTree(cc1_coords)

        for j, cc2 in enumerate(top_components[i+1:], start=i+1):
            cc2_nodes = list(cc2)
            cc2_coords = get_component_coords(cc2)

            # Find minimum distance between components
            min_dist = float('inf')
            min_pair = None

            for k, coord in enumerate(cc2_coords):
                dist, idx = cc1_tree.query(coord)
                if dist < min_dist:
                    min_dist = dist
                    min_pair = (cc1_nodes[idx], cc2_nodes[k])

            if min_pair and min_dist < 500:  # Within 500m
                n1, n2 = min_pair
                c1 = node_coords[n1]
                c2 = node_coords[n2]
                lon1, lat1 = transformer.transform(c1[0], c1[1])
                lon2, lat2 = transformer.transform(c2[0], c2[1])

                gaps.append({
                    'comp1': i+1, 'comp2': j+1,
                    'distance': min_dist,
                    'point1': (lat1, lon1),
                    'point2': (lat2, lon2),
                    'midpoint': ((lat1+lat2)/2, (lon1+lon2)/2)
                })

    # Sort gaps by distance
    gaps.sort(key=lambda x: x['distance'])

    print(f"\n   Found {len(gaps)} potential connections:")
    for gap in gaps[:20]:
        print(f"\n   Gap between Component {gap['comp1']} and {gap['comp2']}:")
        print(f"     Distance: {gap['distance']:.0f}m")
        print(f"     Point 1: {gap['point1'][0]:.5f}, {gap['point1'][1]:.5f}")
        print(f"     Point 2: {gap['point2'][0]:.5f}, {gap['point2'][1]:.5f}")
        print(f"     Midpoint: {gap['midpoint'][0]:.5f}, {gap['midpoint'][1]:.5f}")

    # Create GeoJSON output for visualization
    print("\n5. Creating gap visualization GeoJSON...")
    features = []
    for i, gap in enumerate(gaps[:30]):
        # Line between gap points
        features.append({
            "type": "Feature",
            "properties": {
                "type": "gap",
                "distance": gap['distance'],
                "components": f"{gap['comp1']}-{gap['comp2']}"
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [gap['point1'][1], gap['point1'][0]],
                    [gap['point2'][1], gap['point2'][0]]
                ]
            }
        })

    geojson = {"type": "FeatureCollection", "features": features}
    with open(script_dir / "bike_network_gaps.geojson", 'w') as f:
        json.dump(geojson, f, indent=2)
    print(f"   Saved {len(features)} gap lines to bike_network_gaps.geojson")

    return gaps


def check_virtual_edge_candidates():
    """Check for bike lanes that could have virtual edges but don't."""
    print("\n" + "=" * 70)
    print("VIRTUAL EDGE CANDIDATE ANALYSIS")
    print("=" * 70)

    areas, roads, layers = load_all_data()
    roads_proj = roads.to_crs(TARGET_CRS)

    # Build road network nodes
    coord_to_node = {}
    node_coords = {}
    node_counter = [0]

    def get_or_create_node(x, y):
        key = (round(x / NODE_TOLERANCE) * NODE_TOLERANCE, round(y / NODE_TOLERANCE) * NODE_TOLERANCE)
        if key in coord_to_node:
            return coord_to_node[key]
        nid = node_counter[0]
        node_counter[0] += 1
        coord_to_node[key] = nid
        node_coords[nid] = (x, y)
        return nid

    for _, row in roads_proj.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or geom.geom_type != 'LineString':
            continue
        coords = list(geom.coords)
        if len(coords) >= 2:
            get_or_create_node(coords[0][0], coords[0][1])
            get_or_create_node(coords[-1][0], coords[-1][1])

    node_ids = list(node_coords.keys())
    coords_arr = np.array([node_coords[n] for n in node_ids])
    tree = cKDTree(coords_arr)

    transformer = Transformer.from_crs(TARGET_CRS, WGS84, always_xy=True)

    print("\n1. Checking bike lanes for nearby road nodes...")

    # Look for bike lanes with endpoints far from road nodes
    far_endpoints = []

    for layer_name, gdf in layers.items():
        if len(gdf) == 0:
            continue

        gdf_proj = gdf.to_crs(TARGET_CRS)

        for idx, row in gdf_proj.iterrows():
            geom = row.geometry
            lane_name = row.get('Name', f'{layer_name}_{idx}')

            if geom is None or geom.is_empty:
                continue

            lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms)

            for line in lines:
                if line.length < 20:
                    continue

                # Check start endpoint
                start = line.coords[0]
                start_dist, _ = tree.query([start[0], start[1]])

                # Check end endpoint
                end = line.coords[-1]
                end_dist, _ = tree.query([end[0], end[1]])

                # If either endpoint is far from road network
                if start_dist > VIRTUAL_EDGE_THRESHOLD or end_dist > VIRTUAL_EDGE_THRESHOLD:
                    lon, lat = transformer.transform(line.centroid.x, line.centroid.y)
                    far_endpoints.append({
                        'layer': layer_name,
                        'name': lane_name,
                        'length': line.length,
                        'start_dist': start_dist,
                        'end_dist': end_dist,
                        'lat': lat,
                        'lon': lon
                    })

    print(f"\n2. Bike lanes with endpoints far (>{VIRTUAL_EDGE_THRESHOLD}m) from road network:")

    if far_endpoints:
        # Group by layer
        for layer_name in layers.keys():
            layer_issues = [e for e in far_endpoints if e['layer'] == layer_name]
            if layer_issues:
                print(f"\n   {layer_name.upper()}:")
                for e in layer_issues[:10]:
                    print(f"     {e['name']} ({e['length']:.0f}m)")
                    print(f"       Start: {e['start_dist']:.0f}m from road, End: {e['end_dist']:.0f}m from road")
                    print(f"       Location: {e['lat']:.5f}, {e['lon']:.5f}")
                if len(layer_issues) > 10:
                    print(f"     ... and {len(layer_issues)-10} more")
    else:
        print("   All bike lane endpoints are within range of road network!")

    return far_endpoints


if __name__ == "__main__":
    # Run all analyses
    issues = analyze_bike_lane_features()
    gaps = find_bike_network_gaps()
    far_eps = check_virtual_edge_candidates()

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"- Lanes with connectivity issues: {len(issues) if issues else 0}")
    print(f"- Gaps between bike network components: {len(gaps) if gaps else 0}")
    print(f"- Lanes with far endpoints: {len(far_eps) if far_eps else 0}")
